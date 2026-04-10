
import os
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Benchmarking Visual Geolocalization",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Training parameters
    parser.add_argument("--train_batch_size", type=int, default=4,
                        help="Number of triplets (query, pos, negs) in a batch. Each triplet consists of 12 images")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.00005, help="_")
    parser.add_argument("--optim", type=str, default="adamW", help="_", choices=["adam", "sgd", "adamW"])
    parser.add_argument("--epochs_num", type=int, default=20,
                        help="number of epochs to train for")
    parser.add_argument("--negs_num_per_query", type=int, default=10,
                        help="How many negatives to consider per each query in the loss")
    parser.add_argument("--neg_samples_num", type=int, default=1000,
                        help="How many negatives to use to compute the hardest ones")
    parser.add_argument("--mining", type=str, default="partial", choices=["partial", "full", "random", "msls_weighted"])
    # Inference parameters
    parser.add_argument("--infer_batch_size", type=int, default=16,
                        help="Batch size for inference (caching and testing)")
    # Model parameters
    parser.add_argument("--backbone", type=str, default="dinov2",
                        choices=["dinov2", "vit", "clip"], help="_")
    parser.add_argument("--backbone_size", type=str, default="b",
                        choices=["s", "b", "l"],
                        help="DINOv2 backbone size: s=vit_small(384), b=vit_base(768), l=vit_large(1024)")
    parser.add_argument("--aggregator", type=str, default= None,
                        choices=["netvlad", "salad", "boq", None])
    parser.add_argument("--freeze_te", type=int, default=None, choices=list(range(0, 11)))
    parser.add_argument("--insert_te", type=int, default=8, choices=list(range(0, 11)))
    parser.add_argument("--trunc_te", type=int, default=None, choices=list(range(0, 11)))
    parser.add_argument("--num_learnable_aggregation_tokens", type=int, default=8)
    parser.add_argument('--fc_output_dim', type=int, default=None,
                        help="Output dimension of fully connected layer. If None, don't use a fully connected layer.")
    # Initialization parameters
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--foundation_model_path", type=str, default=None,
                        help="Path to load foundation model checkpoint.")  
    parser.add_argument("--cache_refresh_rate", type=int, default=1000,
                        help="How many queries are selected each cache refresh (ms2_train only)")
    parser.add_argument("--queries_per_epoch", type=int, default=2000,
                        help="Total number of queries sampled per training epoch (ms2_train only)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to load checkpoint from, for resuming training or testing.")
    parser.add_argument("--finetune", type=str, default=None,
                        help="Path to a model-only checkpoint (e.g. ImAge_GSV.pth) to finetune from. "
                             "Loads weights only; optimizer and epoch state start fresh.")
    
    # Other parameters
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num_workers", type=int, default=8, help="num_workers for all dataloaders")
    parser.add_argument('--resize', type=int, default=[224, 224], nargs=2, help="Resizing shape for images (HxW).") # 322 x 322
    parser.add_argument('--test_method', type=str, default="hard_resize",
                        choices=["hard_resize", "single_query", "central_crop", "five_crops", "nearest_crop", "maj_voting"],
                        help="This includes pre/post-processing methods and prediction refinement")
    parser.add_argument("--majority_weight", type=float, default=0.01, 
                        help="only for majority voting, scale factor, the higher it is the more importance is given to agreement")
    parser.add_argument("--val_positive_dist_threshold", type=int, default=10, help="_")
    parser.add_argument("--train_positives_dist_threshold", type=int, default=10, help="_")
    parser.add_argument('--recall_values', type=int, default=[1, 5, 10, 100], nargs="+",
                        help="Recalls to be computed, such as R@5.")
    # Data augmentation parameters
    parser.add_argument("--brightness", type=float, default=None, help="_")
    parser.add_argument("--contrast", type=float, default=None, help="_")
    parser.add_argument("--saturation", type=float, default=None, help="_")
    parser.add_argument("--hue", type=float, default=None, help="_")
    parser.add_argument("--rand_perspective", type=float, default=None, help="_")
    parser.add_argument("--horizontal_flip", action='store_true', help="_")
    parser.add_argument("--random_resized_crop", type=float, default=None, help="_")
    parser.add_argument("--random_rotation", type=float, default=None, help="_")
    # Paths parameters
    parser.add_argument("--datasets_folder", type=str, default=None, help="Path with all datasets")
    parser.add_argument("--dataset_name", type=str, default="pitts30k", help="Relative path of the dataset")

    parser.add_argument("--save_dir", type=str, default="logs",
                        help="Folder name of the current run (saved in ./logs/)")
    parser.add_argument("--train_seq", type=str, default="none", help="_", nargs="+", choices=["Campus", "Residential", "Urban", 'KAIST', 'SNU', 'Valley', 'r0', 'r1'])
    parser.add_argument("--test_seq", type=str, default="none", help="_", nargs="+", choices=["Campus", "Residential", "Urban", 'KAIST', 'SNU', 'Valley', 'r0', 'r1'])
    parser.add_argument("--img_time", type=str, default="allday",
                        choices=["allday", "daytime", "nighttime", "latetime"])
    parser.add_argument("--rgb_model_path", type=str, default="/home/jwkim/workspace/benchmark_THR2RGB/ImAge/ImAge_GSV.pth")
    parser.add_argument("--comment", type=str, default="default")
    
    # Gap alignment (Method: Procrustes subspace loss)
    parser.add_argument("--gap_lambda",    type=float, default=0.1,
                        help="Weight for gap alignment loss. 0 = disabled.")
    parser.add_argument("--gap_k",         type=int,   default=20,
                        help="Procrustes SVD top-k directions for gap alignment subspace.")
    parser.add_argument("--gap_max_pairs", type=int,   default=2000,
                        help="Max matched pairs accumulated per epoch for GapAligner update.")

    args = parser.parse_args()
    
    return args

