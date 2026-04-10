# import torchvision; torchvision.utils.save_image(tensor_image.cpu().detach(), 'debug_img.png')
import custom_wandb as cw
import torch
import logging
import random
import numpy as np
from tqdm import tqdm,trange
import multiprocessing
from os.path import join
from datetime import datetime
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
torch.backends.cudnn.benchmark= True  # Provides a speedup

import loss
import util
import test
import math
import parser
import commons
import network
import datasets_ws
from gap_aligner import GapAligner
from visualize_retrieval import visualize_epoch

import warnings
warnings.filterwarnings("ignore")

BCS_loss = loss.BCS(lmbd=10)
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    
def train_model(args):
    #### Creation of Datasets
    logging.debug(f"Loading dataset {args.dataset_name} from folder {args.datasets_folder}")

    ############################################################
    args.sequences = args.train_seq
    lejepa_ds = datasets_ws.LeJEPADataset(args, args.datasets_folder, args.dataset_name, "train", args.negs_num_per_query)
    logging.info(f"Train query set: {lejepa_ds}")

    test_sequences = args.test_seq
    val_ds_list = []
    test_ds_list = []

    for seq in test_sequences:
        args.sequences = [seq]  
        
        val_ds0 = datasets_ws.BaseDataset(args, args.datasets_folder, args.dataset_name, "test")
        val_ds_list.append(val_ds0)
        logging.info(f"[Val - {seq}] Database: {val_ds0.database_num}, Queries: {val_ds0.queries_num}, Total: {len(val_ds0)}")
        
        val_ds1 = datasets_ws.BaseDataset(args, args.datasets_folder, args.dataset_name, "test")
        test_ds_list.append(val_ds1)
        logging.info(f"[Test - {seq}] Database: {val_ds1.database_num}, Queries: {val_ds1.queries_num}, Total: {len(val_ds1)}")

    args.sequences = args.train_seq
    ############################################################

    #### Initialize model
    model = network.VPRmodel(args)
    model = model.to(args.device)
    model = torch.nn.DataParallel(model)

    #### Print the number of model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    aggregator_params = sum(p.numel() for p in model.module.aggregator.parameters()) if model.module.aggregator else 0
    adapter_params = sum(p.numel() for n, p in model.module.backbone.named_parameters() if 'adapter' in n)

    print(f"The entire parameters: {total_params / 1e6:.2f}M")
    print(f"The trainable parameters: {trainable_params / 1e6:.2f}M")
    print(f"The aggregator parameters: {aggregator_params / 1e6:.2f}M")
    print(f"The adapter parameters: {adapter_params / 1e6:.2f}M")

    #### Initialize agg tokens
    if not args.aggregator:
        args.features_dim = model.module.backbone.embed_dim * args.num_learnable_aggregation_tokens

    if args.aggregator in ["netvlad"]:  # If using NetVLAD layer, initialize it
        args.features_dim = 768
        if not args.resume:
            lejepa_ds = datasets_ws.LeJEPADataset(args, args.datasets_folder, "msls", "train", args.negs_num_per_query)
            logging.info(f"Train query set: {lejepa_ds}")
            lejepa_ds.is_inference = True
            pretrained_model, _ = network.get_backbone(args)
            model.module.agg.initialize_netvlad_layer(args, lejepa_ds, pretrained_model.to(args.device)) 
        args.features_dim = args.features_dim * 8

    logging.info(f"Output dimension of the model is {args.features_dim}")

    #### Setup Optimizer and Loss
    if args.optim == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.001)
    elif args.optim == "adamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-2)

    #### Gap Aligner (Procrustes subspace alignment, Method 1)
    gap_aligner = GapAligner(k=args.gap_k, pca_dim=256) if args.gap_lambda > 0 else None
    if gap_aligner is not None:
        logging.info(f"GapAligner enabled: lambda={args.gap_lambda}, k={args.gap_k}, max_pairs={args.gap_max_pairs}")


    #### Resume model, optimizer, and other training parameters
    if args.resume:
        model, optimizer, best_r1_r5, start_epoch_num, not_improved_num = util.resume_train(args, model, optimizer)
        logging.info(f"Resuming from epoch {start_epoch_num} with best (R@1 + R@5) {best_r1_r5:.1f}")
    else:
        best_r1_r5 = start_epoch_num = not_improved_num = 0

    #### GapAligner / LDAProjector pre-init
    if gap_aligner is not None:
        logging.info("Pre-initializing GapAligner")
        lejepa_ds.is_inference = True
        lejepa_ds.compute_triplets(args, model)
        lejepa_ds.is_inference = False

        init_dl = DataLoader(dataset=lejepa_ds, num_workers=args.num_workers,
                            batch_size=args.train_batch_size,
                            collate_fn=datasets_ws.collate_fn,
                            pin_memory=(args.device == "cuda"),
                            drop_last=True)

        init_thr, init_rgb, n_init = [], [], 0
        model.eval()
        with torch.no_grad():
            for images, pos_img, _ in init_dl:
                if n_init >= args.gap_max_pairs:
                    break
                is_thermal = torch.zeros(images.shape[0]+pos_img.shape[0], dtype=torch.bool)
                for i in range(images.shape[0]):
                    is_thermal[i] = True
                for i in range(pos_img.shape[0]):
                    is_thermal[images.shape[0]:images.shape[0]+i] = False
                descriptors = model(torch.cat([images, pos_img], dim=0).to(args.device), is_thermal=is_thermal.to(args.device))
                init_thr.append(descriptors[:images.shape[0]].float().cpu())
                init_rgb.append(descriptors[images.shape[0]:].float().cpu())
                n_init += images.shape[0]

        if gap_aligner is not None:
            logging.info(f"  [GapAligner] Pre-init with {n_init} pairs")
            gap_aligner.update(torch.cat(init_rgb).numpy(), torch.cat(init_thr).numpy())
            
    #### Training loop
    loops_num = math.ceil(args.queries_per_epoch / args.cache_refresh_rate)

    thermal_flag = torch.ones(1, dtype=torch.bool)
    rgb_flags = torch.zeros(1 + args.negs_num_per_query, dtype=torch.bool)
    bundle_flags = torch.cat([thermal_flag, rgb_flags]).to(args.device)  
    flags = bundle_flags.repeat(args.train_batch_size)

    for epoch_num in range(start_epoch_num, args.epochs_num):
        logging.info(f"Start training epoch: {epoch_num:02d}")
        epoch_start_time = datetime.now()
        epoch_losses = []
        epoch_inv_losses = []
        epoch_sig_losses = []
        epoch_gap_cosines = []
        epoch_gap_losses  = []
        epoch_cosines = []
        gap_buf_thr, gap_buf_rgb, gap_n_pairs = [], [], 0
        
        for loop_num in range(loops_num):
            logging.debug(f"Cache refresh: {loop_num + 1} / {loops_num}")

            lejepa_ds.is_inference = True
            lejepa_ds.compute_triplets(args, model)
            lejepa_ds.is_inference = False

            triplets_dl = DataLoader(dataset=lejepa_ds, num_workers=args.num_workers,
                                    batch_size=args.train_batch_size,
                                    collate_fn=datasets_ws.collate_fn,
                                    pin_memory=(args.device == "cuda"),
                                    drop_last=True)

            model = model.train()
            import torchvision; 
            for batch in tqdm(triplets_dl, ncols=100):
                images, pos_img, views = batch
                optimizer.zero_grad()

                ####################################
                ########## LeJEPA Sig Loss #########
                B = images.shape[0]
                views_flat = views.view(B * views.shape[1], *views.shape[2:])  # [B*N, 3, H, W]

                with torch.no_grad():
                    x_rgb = model(pos_img.to(args.device), is_thermal=False)
                x_ir = model(torch.cat([images, views_flat], dim=0).to(args.device), is_thermal=True)
                x_ir_global = x_ir[:args.train_batch_size]
                
                pos_z = model.module.projector(x_rgb)
                ir_z = model.module.projector(x_ir)
                
                ir_global = ir_z[:args.train_batch_size]
                all_z = torch.cat([pos_z, ir_z])
                num_views = ir_z.size(0) // pos_z.size(0)
                pos_z_targets = pos_z.repeat(num_views, 1)
                inv_loss = F.mse_loss(pos_z_targets, ir_z).mean()

                sigreg_loss = BCS_loss(all_z, return_sim=False, return_sigreg=True)
                sigreg_loss = sigreg_loss['bcs_loss']
                BCS_loss.step += 1

                loss = inv_loss + 10 * sigreg_loss
                ####################################
                ######## Gap alignment loss ########
                # loss_gap = torch.tensor(0.0, device=args.device)
                # if gap_aligner is not None:
                #     # x_rgb         : [B, 6144]
                #     # x_ir_global   : [B, 6144]
                #     loss_gap = gap_aligner.loss(x_ir_global, x_rgb.detach())
                #     loss += args.gap_lambda * loss_gap
                    
                #     with torch.no_grad():
                #         cos = torch.nn.functional.cosine_similarity(
                #             x_ir_global, x_rgb, dim=-1).mean().item()
                #         epoch_gap_cosines.append(cos)

                # # Gap buffer 누적 (epoch-end GapAligner.update() 용)
                # if gap_aligner is not None and gap_n_pairs < args.gap_max_pairs:
                #     gap_buf_thr.append(x_ir_global.cpu())
                #     gap_buf_rgb.append(x_rgb.detach().cpu())
                #     gap_n_pairs += x_ir_global.shape[0]
                ####################################

                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())
                epoch_inv_losses.append(inv_loss.item())
                epoch_sig_losses.append(sigreg_loss.item())

                with torch.no_grad():
                    cos = F.cosine_similarity(pos_z, ir_global, dim=-1).mean().item()
                    epoch_cosines.append(cos)
                    
                # if gap_aligner is not None:
                #     epoch_gap_losses.append(loss_gap.item())
                del loss

            logging.debug(f"Epoch[{epoch_num:02d}]({loop_num + 1}/{loops_num}): "
                        f"avg loss={np.mean(epoch_losses):.4f}  "
                        f"inv={np.mean(epoch_inv_losses):.4f}  "
                        f"sig={np.mean(epoch_sig_losses):.4f} "
                        f"gap={np.mean(epoch_gap_losses):.4f} "
                        f"matched_cos={np.mean(epoch_gap_cosines):.4f}" if gap_aligner else "")

        logging.info(f"Finished epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
                    f"avg loss={np.mean(epoch_losses):.4f}  "
                    f"inv={np.mean(epoch_inv_losses):.4f}  "
                    f"sig={np.mean(epoch_sig_losses):.4f}  "
                    f"gap={np.mean(epoch_gap_losses):.4f} "
                    f"matched_cos={np.mean(epoch_gap_cosines):.4f}" if gap_aligner else "")


        # ── Gap aligner epoch-end update & analysis ────────────────
        if gap_aligner is not None:
            avg_gap_loss = np.mean(epoch_gap_losses) if epoch_gap_losses else 0.0
            avg_gap_cos  = np.mean(epoch_gap_cosines) if epoch_gap_cosines else 0.0
            cw.wandb_log("loss", "total", np.mean(epoch_losses))
            cw.wandb_log("loss", "invariance", np.mean(epoch_inv_losses))
            cw.wandb_log("loss", "sigreg", np.mean(epoch_sig_losses))
            cw.wandb_log("lejepa", "matched_cos", np.mean(epoch_cosines))
            
            cw.wandb_log("loss", "gap",         avg_gap_loss)
            cw.wandb_log("gap",  "matched_cos", avg_gap_cos)
            cw.wandb_log("gap",  "n_pairs",     gap_n_pairs)

            if gap_buf_thr:
                f_thr_np = torch.cat(gap_buf_thr).detach().cpu().numpy()
                f_rgb_np = torch.cat(gap_buf_rgb).detach().cpu().numpy()
                logging.info(f"  [GapAligner] Updating with {gap_n_pairs} pairs...")
                gap_aligner.update(f_rgb_np, f_thr_np)
                # sigma_k 상위 5개 wandb 로그
                for i, s in enumerate(gap_aligner.sigma_k[:5].tolist()):
                    cw.wandb_log("gap", f"sigma_{i}", s)
                
        # Compute recalls on all validation sequences
        all_r1, all_r5 = [], []
        for seq, val_dataset in zip(test_sequences, val_ds_list):
            recalls, recalls_str, predictions, positives_per_query = test.test(
                args, val_dataset, model, return_predictions=True)
            logging.info(f"Recalls on [{seq}] {val_dataset}: {recalls_str}")
            vis_dir = join(args.save_dir, "vis")
            visualize_epoch(val_dataset, predictions, positives_per_query,
                            epoch_num, seq, vis_dir, n_samples=10)
            cw.wandb_log("r1", seq, recalls[0])
            cw.wandb_log("r5", seq, recalls[1])
            cw.wandb_log("r10", seq, recalls[2])
            all_r1.append(recalls[0])
            all_r5.append(recalls[1])

        avg_r1_r5 = np.mean(all_r1) + np.mean(all_r5)
        is_best = avg_r1_r5 > best_r1_r5

        util.save_checkpoint(args, {"epoch_num": epoch_num, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "recalls": (np.mean(all_r1), np.mean(all_r5)), "best_r1_r5": best_r1_r5,
            "not_improved_num": not_improved_num
        }, is_best, filename="last_model.pth")

        if is_best:
            logging.info(f"Improved: previous best avg (R@1 + R@5) = {best_r1_r5:.1f}, current = {avg_r1_r5:.1f}")
            best_r1_r5 = avg_r1_r5
            not_improved_num = 0
        else:
            not_improved_num += 1
            logging.info(f"Not improved: {not_improved_num} / {args.patience}: best = {best_r1_r5:.1f}, current = {avg_r1_r5:.1f}")
            if not_improved_num >= args.patience:
                logging.info(f"Performance did not improve for {not_improved_num} epochs. Stop training.")
                break

    logging.info(f"Best avg (R@1 + R@5): {best_r1_r5:.1f}")
    logging.info(f"Trained for {epoch_num+1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")

    # Test best model on all test sequences
    logging.info("Test *best* model on all test sequences")
    best_model_state_dict = torch.load(join(args.save_dir, "best_model.pth"), weights_only=False)["model_state_dict"]
    model.load_state_dict(best_model_state_dict)
    for seq, test_dataset in zip(test_sequences, test_ds_list):
        recalls, recalls_str = test.test(args, test_dataset, model, test_method=args.test_method)
        logging.info(f"Recalls on [{seq}] {test_dataset}: {recalls_str}")
        
if __name__ == "__main__":
    #### Initial setup: parser, logging...
    set_seed()
    args = parser.parse_arguments()
    cw.wandb_init(args, name=args.comment)
    start_time = datetime.now()
    args.save_dir = join(args.save_dir, args.comment, util.get_timestamp())
    
    commons.setup_logging(args.save_dir)
    util.save_files(args.save_dir)
    
    commons.make_deterministic(args.seed)
    logging.info(f"Arguments: {args}")
    logging.info(f"The outputs are being saved in {args.save_dir}")
    logging.info(f"Using {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs")

    train_model(args)
    