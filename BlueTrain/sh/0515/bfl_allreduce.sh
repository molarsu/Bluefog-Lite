# for pytorch bfl
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bflrun -np 8 python BlueTrain/main.py \
    --model vit_b \
    --output-path ./outputs/0515/bluefog/vit_b_8gpu/all_reduce \
    --dist-mode bluefog \
    --dist-optimizer allreduce \
    --backend nccl \
    --log-interval 1 \
    --batch-size 64 \
    --epochs 100 \
    --lr 0.0125 \
    --momentum 0.9 \
    --world-size 8 \

# for pytorch bfl
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bflrun -np 8 python BlueTrain/main.py \
    --model vit_b \
    --output-path ./outputs/0515/bluefog/vit_b_8gpu/all_reduce \
    --dist-mode bluefog \
    --dist-optimizer gradient_allreduce \
    --backend nccl \
    --log-interval 1 \
    --batch-size 64 \
    --epochs 100 \
    --lr 0.0125 \
    --momentum 0.9 \
    --world-size 8

# for pytorch bfl
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bflrun -np 8 python BlueTrain/main.py \
    --model vit_b \
    --output-path ./outputs/0515/bluefog/vit_b_8gpu/all_reduce \
    --dist-mode bluefog \
    --dist-optimizer neighbor_allreduce \
    --backend nccl \
    --log-interval 1 \
    --batch-size 64 \
    --epochs 100 \
    --lr 0.0125 \
    --momentum 0.9 \
    --world-size 8