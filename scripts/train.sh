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

# # Vanilla adapter (GPU 5)
# OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=5 python3 train.py \
#     $COMMON_ARGS \
#     --comment "vanilla-adapter" \
#     > logs/vanilla-adapter.log 2>&1 &
# echo "[vanilla-adapter] PID=$!"

# # FiLM shared adapter (GPU 6)
# OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=6 python3 train.py \
#     $COMMON_ARGS \
#     --film_adapter \
#     --comment "film-adapter" \
#     > logs/film-adapter.log 2>&1 &
# echo "[film-adapter]    PID=$!"

# FiLM + Flow matching loss v2 (GPU 7)
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=6 python3 train.py \
    $COMMON_ARGS \
    --film_adapter \
    --lambda_flow 0.1 \
    --comment "film-flow-v4_normtarin" \
    > logs/film-flow-v4_normtarin.log 2>&1 &
echo "[film-flow-v4_normtarin]    PID=$!"

wait
echo "All experiments finished."
