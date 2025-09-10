torchrun --standalone --nproc_per_node=gpu model/TC_predict/multi_gpu_train.py \
    --epochs 30 \
    --batch_size 8 \
    --lr 0.00001 \
    --loss_fn huber_smooth \
    --use_rgb \
    --use_modulation \
    --checkpoint_interval 1 \
    --log_dir logs/origin \
    --checkpoint_dir checkpoints/origin

torchrun --standalone --nproc_per_node=gpu model/TC_predict/multi_gpu_train.py \
    --epochs 30 \
    --batch_size 8 \
    --lr 0.00001 \
    --loss_fn huber_smooth \
    --checkpoint_interval 1 \
    --log_dir logs/no_rgb \
    --checkpoint_dir checkpoints/no_rgb

torchrun --standalone --nproc_per_node=gpu model/TC_predict/multi_gpu_train.py \
    --epochs 30 \
    --batch_size 8 \
    --lr 0.00001 \
    --loss_fn huber_smooth \
    --use_rgb \
    --checkpoint_interval 1 \
    --log_dir logs/no_modulation \
    --checkpoint_dir checkpoints/no_modulation

torchrun --standalone --nproc_per_node=gpu model/TC_predict/multi_gpu_train.py \
    --epochs 30 \
    --batch_size 8 \
    --lr 0.00001 \
    --loss_fn huber_smooth \
    --use_modulation \
    --checkpoint_interval 1 \
    --log_dir logs/pc_with_no_rgb \
    --checkpoint_dir checkpoints/pc_with_no_rgb