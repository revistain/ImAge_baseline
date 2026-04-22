COMMON_ARGS="
    --datasets_folder=/home/jwkim/workspace/benchmark_THR2RGB/Datasets
    --dataset_name=ms2dataset
    --backbone=dinov2
    --num_learnable_aggregation_tokens=8
    --train_batch_size=4
    --lr=0.00005
    --train_seq Urban
    --test_seq Campus Residential
    --epochs_num=40
    --patience=20
    --cache_refresh_rate=1000
    --queries_per_epoch=2000
    --margin=0.1
    --insert_te=8
    --foundation_model_path=/home/jwkim/workspace/benchmark_THR2RGB/pretrained/dinov2_vitb14_reg4_pretrain.pth
"

# CFM: triplet only (ablation — no FM loss)
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=5 python3 train.py \
    $COMMON_ARGS \
    --lambda_flow 0.0 \
    --comment "cfm-triplet" \
    > logs/cfm-triplet.log 2>&1 &
echo "[cfm-triplet] PID=$!"

# CFM: triplet + OT-CFM loss (lambda=0.1)
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=4 python3 train.py \
    $COMMON_ARGS \
    --lambda_flow 0.1 \
    --comment "cfm-fm" \
    > logs/cfm-fm.log 2>&1 &
echo "[cfm-fm]      PID=$!"

wait
echo "All experiments finished."
