OMP_NUM_THREADS=4 \
CUDA_VISIBLE_DEVICES=6 \
python3 train.py \
    --datasets_folder=/home/jwkim/workspace/benchmark_THR2RGB/Datasets \
    --dataset_name=ms2dataset \
    --backbone=dinov2 \
    --num_learnable_aggregation_tokens=8 \
    --train_batch_size=4 \
    --lr=0.00005 \
    --train_seq r0 \
    --test_seq Campus Residential \
    --epochs_num=40 \
    --patience=20 \
    --cache_refresh_rate=1000 \
    --queries_per_epoch=2000 \
    --margin=0.1 \
    --insert_te=8 \
    --gap_lambda 0.1 \
    --foundation_model_path=/home/jwkim/workspace/benchmark_THR2RGB/pretrained/dinov2_vitb14_reg4_pretrain.pth

# Scene 종류 : ['Campus', 'Residential', 'Urban', 'KAIST', 'SNU', 'Valley', 'r0', 'r1'] 