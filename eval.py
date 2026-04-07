
import os
import torch
import parser
import logging
from os.path import join
from datetime import datetime

import test
import util
import commons
import datasets_ws
import network
import warnings
warnings.filterwarnings("ignore")

######################################### SETUP #########################################
args = parser.parse_arguments()
start_time = datetime.now()
args.save_dir = join("test", args.save_dir, start_time.strftime('%Y-%m-%d_%H-%M-%S'))
commons.setup_logging(args.save_dir)
commons.make_deterministic(args.seed)

logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.save_dir}")
######################################### MODEL #########################################
model = network.VPRmodel(args)
model = model.to(args.device)

if args.resume is not None:
    logging.info(f"Resuming model from {args.resume}")
    model = util.resume_model(args, model)

# Enable DataParallel after loading checkpoint, otherwise doing it before
# would append "module." in front of the keys of the state dict triggering errors
model = torch.nn.DataParallel(model)

if args.aggregator:
    pass  # features_dim already set inside get_aggregator()
else:
    args.features_dim = args.num_learnable_aggregation_tokens * model.module.backbone.embed_dim
logging.info(f"features_dim: {args.features_dim}")

test_ds_list = []
test_sequences = args.test_seq
for seq in test_sequences:
    args.sequences = [seq]  
    ######################################### DATASETS #########################################
    test_ds = datasets_ws.BaseDataset(args, args.datasets_folder, args.dataset_name, "test")
    test_ds_list.append(test_ds)
    logging.info(f"Test set: {test_ds}")

######################################### TEST on TEST SET #########################################
for seq, val_dataset in zip(test_sequences, test_ds_list):
    recalls, recalls_str = test.test(args, val_dataset, model, args.test_method)
    logging.info(f"Recalls on {test_ds}: {recalls_str}")

    logging.info(f"Finished in {str(datetime.now() - start_time)[:-7]}")