OMP_NUM_THREADS=8 \
CUDA_VISIBLE_DEVICES=5 \
python3 eval.py \
--datasets_folder=/home/jwkim/workspace/benchmark_THR2RGB/Datasets \
--test_seq Campus Residential Urban \
--backbone=dinov2 \
--freeze_te=8 \
--num_learnable_aggregation_tokens=8 \
--resume=/home/jwkim/workspace/benchmark_THR2RGB/pretrained/best_model.pth