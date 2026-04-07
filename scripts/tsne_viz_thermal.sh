OMP_NUM_THREADS=4 \
CUDA_VISIBLE_DEVICES=3 \
python3 tsne_viz.py \
--datasets_folder=/home/jwkim/workspace/benchmark_THR2RGB/Datasets \
--test_seq Campus Residential Urban \
--backbone=dinov2 \
--backbone_size=s \
--freeze_te=8 \
--num_learnable_aggregation_tokens=8 \
--resume=/home/jwkim/workspace/benchmark_THR2RGB/ImAge/logs/default/2026-03-09_05-12-44/best_model.pth
