OMP_NUM_THREADS=4 \
CUDA_VISIBLE_DEVICES=2 \
python3 train.py \
--datasets_folder=/home/jwkim/workspace/benchmark_THR2RGB/Datasets \
--train_seq Campus \
--test_seq Campus Residential Urban \
--backbone=dinov2 \
--backbone_size=b \
--freeze_te=8 \
--num_learnable_aggregation_tokens=8 \
--resume=/home/jwkim/workspace/benchmark_THR2RGB/ImAge/ImAge_GSV.pth \
--lr=0.00001 \
--epochs_num=20 \
--train_batch_size=4 \
--patience=10
