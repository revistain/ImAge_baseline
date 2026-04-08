# import torchvision; torchvision.utils.save_image(tensor_image.cpu().detach(), 'debug_img.png')
import custom_wandb as cw
import torch
import logging
import numpy as np
from tqdm import tqdm,trange
import multiprocessing
from os.path import join
from datetime import datetime
from torch.utils.data.dataloader import DataLoader
torch.backends.cudnn.benchmark= True  # Provides a speedup

import util
import test
import math
import parser
import commons
import network
import datasets_ws
from gap_aligner import GapAligner
from torch.cuda.amp import GradScaler, autocast

import warnings
warnings.filterwarnings("ignore")

#### Initial setup: parser, logging...
args = parser.parse_arguments()
cw.wandb_init(args, name="ImAge-DINOv2_b-MutualRGBInit-thermalAdapter(agg_frozen)")
start_time = datetime.now()
args.save_dir = join("logs", args.save_dir, start_time.strftime('%Y-%m-%d_%H-%M-%S'))
commons.setup_logging(args.save_dir)
commons.make_deterministic(args.seed)
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.save_dir}")
logging.info(f"Using {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs")

#### Creation of Datasets
logging.debug(f"Loading dataset {args.dataset_name} from folder {args.datasets_folder}")

############################################################
args.sequences = args.train_seq
triplets_ds = datasets_ws.TripletsDataset(args, args.datasets_folder, args.dataset_name, "train", args.negs_num_per_query)
logging.info(f"Train query set: {triplets_ds}")

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
adapter_params = sum(p.numel() for n, p in model.module.backbone_thermal.named_parameters() if 'adapter' in n)

print(f"The entire parameters: {total_params / 1e6:.2f}M")
print(f"The trainable parameters: {trainable_params / 1e6:.2f}M")
print(f"The aggregator parameters: {aggregator_params / 1e6:.2f}M")
print(f"The adapater parameters: {adapter_params / 1e6:.2f}M")

#### Initialize agg tokens
if not args.aggregator:
    args.features_dim = model.module.backbone_rgb.embed_dim * args.num_learnable_aggregation_tokens

if args.aggregator in ["netvlad"]:  # If using NetVLAD layer, initialize it
    args.features_dim = 768
    if not args.resume:
        triplets_ds = datasets_ws.TripletsDataset(args, args.datasets_folder, "msls", "train", args.negs_num_per_query)
        logging.info(f"Train query set: {triplets_ds}")
        triplets_ds.is_inference = True
        pretrained_model = network.get_backbone(args)
        model.module.agg.initialize_netvlad_layer(args, triplets_ds, pretrained_model.to(args.device))
    args.features_dim = args.features_dim * 8

logging.info(f"Output dimension of the model is {args.features_dim}")

#### Setup Optimizer and Loss
if args.optim == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
elif args.optim == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.001)

#### Gap Aligner (Procrustes subspace alignment)
gap_aligner = GapAligner(k=args.gap_k, pca_dim=256) if args.gap_lambda > 0 else None
if gap_aligner is not None:
    logging.info(f"GapAligner enabled: lambda={args.gap_lambda}, k={args.gap_k}, max_pairs={args.gap_max_pairs}")

#### Resume model, optimizer, and other training parameters
if args.resume:
    model, optimizer, best_r1_r5, start_epoch_num, not_improved_num = util.resume_train(args, model, optimizer)
    logging.info(f"Resuming from epoch {start_epoch_num} with best (R@1 + R@5) {best_r1_r5:.1f}")
else:
    best_r1_r5 = start_epoch_num = not_improved_num = 0

#### GapAligner pre-init (epoch 0에도 gap loss 적용)
if gap_aligner is not None:
    logging.info("Pre-initializing GapAligner before epoch 0...")
    triplets_ds.is_inference = True
    triplets_ds.compute_triplets(args, model)
    triplets_ds.is_inference = False

    init_dl = DataLoader(dataset=triplets_ds, num_workers=args.num_workers,
                         batch_size=args.train_batch_size,
                         collate_fn=datasets_ws.collate_fn,
                         pin_memory=(args.device == "cuda"),
                         drop_last=True)

    thermal_flag = torch.ones(1, dtype=torch.bool)
    rgb_flags = torch.zeros(1 + args.negs_num_per_query, dtype=torch.bool)
    bundle_flags = torch.cat([thermal_flag, rgb_flags]).to(args.device)
    flags = bundle_flags.repeat(args.train_batch_size)

    init_thr, init_rgb, n_init = [], [], 0
    model.eval()
    with torch.no_grad():
        for images, triplets_local_indexes, _ in init_dl:
            if n_init >= args.gap_max_pairs:
                break
            descriptors = model(images.to(args.device), flags)
            triplets_local_indexes = torch.transpose(
                triplets_local_indexes.view(args.train_batch_size, args.negs_num_per_query, 3), 1, 0)
            q_idx = triplets_local_indexes[0, :, 0]
            p_idx = triplets_local_indexes[0, :, 1]
            init_thr.append(descriptors[q_idx].float().cpu())
            init_rgb.append(descriptors[p_idx].float().cpu())
            n_init += q_idx.shape[0]

    logging.info(f"  [GapAligner] Pre-init with {n_init} pairs")
    gap_aligner.update(torch.cat(init_rgb).numpy(), torch.cat(init_thr).numpy())

#### Training loop (MS2 triplets)
GlobalTriplet = torch.nn.TripletMarginLoss(margin=args.margin, p=2, reduction="sum")
loops_num = math.ceil(args.queries_per_epoch / args.cache_refresh_rate)

thermal_flag = torch.ones(1, dtype=torch.bool)
rgb_flags = torch.zeros(1 + args.negs_num_per_query, dtype=torch.bool)
bundle_flags = torch.cat([thermal_flag, rgb_flags]).to(args.device)
flags = bundle_flags.repeat(args.train_batch_size)

for epoch_num in range(start_epoch_num, args.epochs_num):
    logging.info(f"Start training epoch: {epoch_num:02d}")
    epoch_start_time = datetime.now()
    epoch_losses = []
    epoch_trip_losses = []
    epoch_gap_losses  = []
    epoch_gap_cosines = []
    gap_buf_thr, gap_buf_rgb, gap_n_pairs = [], [], 0

    for loop_num in range(loops_num):
        logging.debug(f"Cache refresh: {loop_num + 1} / {loops_num}")

        triplets_ds.is_inference = True
        triplets_ds.compute_triplets(args, model)
        triplets_ds.is_inference = False

        triplets_dl = DataLoader(dataset=triplets_ds, num_workers=args.num_workers,
                                 batch_size=args.train_batch_size,
                                 collate_fn=datasets_ws.collate_fn,
                                 pin_memory=(args.device == "cuda"),
                                 drop_last=True)

        model = model.train()
        for images, triplets_local_indexes, _ in tqdm(triplets_dl, ncols=100):
            optimizer.zero_grad()

            descriptors = model(images.to(args.device), flags)
            triplets_local_indexes = torch.transpose(
                triplets_local_indexes.view(args.train_batch_size, args.negs_num_per_query, 3), 1, 0)

            # Triplet loss (original)
            loss_trip = 0
            for triplets in triplets_local_indexes:
                queries_indexes, positives_indexes, negatives_indexes = triplets.T
                loss_trip += GlobalTriplet(descriptors[queries_indexes],
                                          descriptors[positives_indexes],
                                          descriptors[negatives_indexes])
            loss_trip /= (args.train_batch_size * args.negs_num_per_query)

            # Gap alignment loss
            # triplets_local_indexes[0]: [BS, 3] — 첫 번째 negative set에서 q/p 인덱스 추출
            # (모든 neg set에서 query·positive 인덱스는 동일)
            loss_gap = torch.tensor(0.0, device=args.device)
            if gap_aligner is not None:
                q_idx = triplets_local_indexes[0, :, 0]
                p_idx = triplets_local_indexes[0, :, 1]
                f_thr = descriptors[q_idx].float()           # gradient 흐름 (thermal)
                f_rgb = descriptors[p_idx].detach().float()   # gradient 차단 (rgb)
                loss_gap = gap_aligner.loss(f_thr, f_rgb)

            loss = loss_trip + args.gap_lambda * loss_gap

            # Gap buffer 누적 (epoch-end GapAligner.update() 용)
            if gap_aligner is not None and gap_n_pairs < args.gap_max_pairs:
                gap_buf_thr.append(f_thr.detach().cpu())
                gap_buf_rgb.append(f_rgb.cpu())
                gap_n_pairs += f_thr.shape[0]

            # Matched-pair cosine (gap 모니터링)
            if gap_aligner is not None:
                with torch.no_grad():
                    cos = torch.nn.functional.cosine_similarity(
                        f_thr, f_rgb, dim=-1).mean().item()
                    epoch_gap_cosines.append(cos)

            del descriptors

            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            epoch_trip_losses.append(loss_trip.item())
            if gap_aligner is not None:
                epoch_gap_losses.append(loss_gap.item())
            del loss

        logging.debug(f"Epoch[{epoch_num:02d}]({loop_num + 1}/{loops_num}): "
                      f"avg trip={np.mean(epoch_trip_losses):.4f}"
                      + (f"  avg gap={np.mean(epoch_gap_losses):.4f}" if gap_aligner else ""))

    logging.info(f"Finished epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
                 f"avg total loss={np.mean(epoch_losses):.4f}  "
                 f"trip={np.mean(epoch_trip_losses):.4f}"
                 + (f"  gap={np.mean(epoch_gap_losses):.4f}  "
                    f"matched_cos={np.mean(epoch_gap_cosines):.4f}" if gap_aligner else ""))

    # ── Gap aligner epoch-end update & analysis ────────────────
    if gap_aligner is not None:
        avg_gap_loss = np.mean(epoch_gap_losses) if epoch_gap_losses else 0.0
        avg_gap_cos  = np.mean(epoch_gap_cosines) if epoch_gap_cosines else 0.0
        cw.wandb_log("loss", "triplet",     np.mean(epoch_trip_losses))
        cw.wandb_log("loss", "gap",         avg_gap_loss)
        cw.wandb_log("gap",  "matched_cos", avg_gap_cos)
        cw.wandb_log("gap",  "n_pairs",     gap_n_pairs)

        if gap_buf_thr:
            f_thr_np = torch.cat(gap_buf_thr).numpy()
            f_rgb_np = torch.cat(gap_buf_rgb).numpy()
            logging.info(f"  [GapAligner] Updating with {gap_n_pairs} pairs...")
            gap_aligner.update(f_rgb_np, f_thr_np)
            # sigma_k 상위 5개 wandb 로그
            for i, s in enumerate(gap_aligner.sigma_k[:5].tolist()):
                cw.wandb_log("gap", f"sigma_{i}", s)

    # Compute recalls on all validation sequences
    all_r1, all_r5 = [], []
    for seq, val_dataset in zip(test_sequences, val_ds_list):
        recalls, recalls_str = test.test(args, val_dataset, model)
        logging.info(f"Recalls on [{seq}] {val_dataset}: {recalls_str}")
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
