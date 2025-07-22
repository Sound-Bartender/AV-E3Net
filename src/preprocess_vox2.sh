export PYTHONPATH=$(pwd):$PYTHONPATH

# 1 2 3 중 택
python preprocess/preprocess_vox2_random.py \
    --voxceleb2_root /development/dataset/voxceleb2 \
    --output_root /development/dataset/voxceleb2_ave3 \
    --n_workers 16 \
    --noise_conditions 1 2 3 \
    --dns_root /development/dataset/DNS-Challenge/noise_fullband/datasets_fullband/noise_fullband
