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
import parser
import commons
import datasets_ws
import network
import math
from torch.cuda.amp import GradScaler, autocast

import warnings
warnings.filterwarnings("ignore")
import os

#### Initial setup: parser, logging...
args = parser.parse_arguments()
cw.wandb_init(args, name="ImAge-DINOv2_b-MutualRGBInit-thermalUnFreezed")
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

# val_ds0 = datasets_ws.BaseDataset(args, args.eval_datasets_folder, "pitts30k", "val")
# logging.info(f"Val set0: {val_ds0}")
# val_ds1 = datasets_ws.BaseDataset(args, args.eval_datasets_folder, "msls", "val")
# logging.info(f"Val set1: {val_ds1}")

#### Initialize model
model = network.VPRmodel(args)
model = model.to(args.device)
model = torch.nn.DataParallel(model)

#### Print the number of model parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
aggregator_params = sum(p.numel() for p in model.module.aggregator.parameters()) if model.module.aggregator else 0

print(f"The entire parameters: {total_params / 1e6:.2f}M")
print(f"The trainable parameters: {trainable_params / 1e6:.2f}M")
print(f"The aggregator parameters: {aggregator_params / 1e6:.2f}M")

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

#### Resume model, optimizer, and other training parameters
if args.resume:
    model, optimizer, best_r1_r5, start_epoch_num, not_improved_num = util.resume_train(args, model, optimizer)
    logging.info(f"Resuming from epoch {start_epoch_num} with best (R@1 + R@5) {best_r1_r5:.1f}")
# elif args.finetune:
#     # 모델 weight만 로드 (epoch/optimizer 상태는 새로 시작)
#     ckpt = torch.load(args.finetune, map_location=args.device, weights_only=False)
#     sd   = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
#     if list(sd.keys())[0].startswith('module.'):
#         sd = {k.replace('module.', ''): v for k, v in sd.items()}
#     model.module.load_state_dict(sd)
#     logging.info(f"Finetuning from {args.finetune} (epoch 0, optimizer reset)")
#     best_r1_r5 = start_epoch_num = not_improved_num = 0
else:
    best_r1_r5 = start_epoch_num = not_improved_num = 0

#### Training loop (MS2 triplets)
GlobalTriplet = torch.nn.TripletMarginLoss(margin=args.margin, p=2, reduction="sum")
loops_num = math.ceil(args.queries_per_epoch / args.cache_refresh_rate)

thermal_flag = torch.zeros(1, dtype=torch.long)
rgb_flags = torch.ones(1 + args.negs_num_per_query, dtype=torch.long)
bundle_flags = torch.cat([thermal_flag, rgb_flags])
flags = bundle_flags.repeat(args.train_batch_size)

for epoch_num in range(start_epoch_num, args.epochs_num):
    logging.info(f"Start training epoch: {epoch_num:02d}")
    epoch_start_time = datetime.now()
    epoch_losses = []

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
            loss = torch.tensor(0.0, device=args.device)
            
            optimizer.zero_grad()
            descriptors = model(images.to(args.device), flags)
            triplets_local_indexes = torch.transpose(
                triplets_local_indexes.view(args.train_batch_size, args.negs_num_per_query, 3), 1, 0)
            for triplets in triplets_local_indexes:
                queries_indexes, positives_indexes, negatives_indexes = triplets.T
                loss += GlobalTriplet(
                    descriptors[queries_indexes],
                    descriptors[positives_indexes],
                    descriptors[negatives_indexes],
                )
            loss /= (args.train_batch_size * args.negs_num_per_query)
            del descriptors

            optimizer.step()
            loss.backward()

            epoch_losses.append(loss.item())
            del loss

        logging.debug(f"Epoch[{epoch_num:02d}]({loop_num + 1}/{loops_num}): "
                      f"average triplet loss = {np.mean(epoch_losses):.4f}")

    logging.info(f"Finished epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
                 f"average epoch triplet loss = {np.mean(epoch_losses):.4f}")

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