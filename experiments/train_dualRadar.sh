export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1

cd src
# train
python mytrain4dual.py \
    ddd \
    --exp_id centerfusion \
    --shuffle_train \
    --train_split train \
    --val_split mini_val \
    --val_intervals 1 \
    --batch_size 1 \
    --lr 2.5e-4 \
    --num_epochs 60 \
    --lr_step 50 \
    --save_point 20,40,50 \
    --not_rand_crop \
    --flip 0.5 \
    --shift 0.1 \
    --pointcloud \
    --pc_z_offset 0.0 \
    --pillar_dims 1.0,0.2,0.2 \
    --max_pc_dist 60.0 \
    --freeze_backbone \
    # --load_model ../models/centernet_baseline_e170.pth \
    # --freeze_backbone \
    # --resume \

cd ..
