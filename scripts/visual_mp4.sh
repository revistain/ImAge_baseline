OMP_NUM_THREADS=4 \
CUDA_VISIBLE_DEVICES=7 \
python3 visualize.py \
    --resume /home/jwkim/workspace/benchmark_THR2RGB/pretrained/best_model.pth \
    --datasets_folder /home/jwkim/workspace/benchmark_THR2RGB/Datasets \
    --dataset_name ms2dataset \
    --backbone dinov2 \
    --num_learnable_aggregation_tokens 8 \
    --test_seq Urban \
    --img_time allday \
    --output_video visualization_Urban_sampling.mp4 \
    --fps 3 \
    --img_time latetime \
    --frame_step 15