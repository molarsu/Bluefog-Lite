# for pytorch bfl
CUDA_VISIBLE_DEVICES=0 \
bflrun -np 1 python BlueTrain/main.py \
    --model vit_b \
    --dist-mode bluefog \
    --backend nccl \
    --log-interval 1 \
    --batch-size 64 \
    --epochs 100 \
    --lr 0.1 \
    --momentum 0.9 \
    --world-size 1 \


# for bluefog