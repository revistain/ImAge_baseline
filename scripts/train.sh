OMP_NUM_THREADS=4 \
CUDA_VISIBLE_DEVICES=5 \
python3 train.py \
    --datasets_folder=/home/jwkim/workspace/benchmark_THR2RGB/Datasets \
    --dataset_name=ms2dataset \
    --backbone=dinov2 \
    --num_learnable_aggregation_tokens=8 \
    --train_batch_size=64 \
    --lr=5e-5 \
    --train_seq r1 \
    --test_seq Campus Residential \
    --epochs_num=50 \
    --patience=20 \
    --cache_refresh_rate=1000 \
    --queries_per_epoch=2000 \
    --insert_te=8 \
    --freeze_te=8 \
    --foundation_model_path=/home/jwkim/workspace/benchmark_THR2RGB/pretrained/dinov2_vitb14_reg4_pretrain.pth \
    --random_resized_crop 0.3 \
    --contrast 1 \
    --comment "ImAge-DINOv2_b-MutualRGBInit-sharedBackbone-thermalAdapter-onlyInvLoss"

# Scene 종류 : ['Campus', 'Residential', 'Urban', 'KAIST', 'SNU', 'Valley', 'r0', 'r1'] 
