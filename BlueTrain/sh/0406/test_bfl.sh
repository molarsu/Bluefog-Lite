# for pytorch bfl
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
bflrun -np 8 python BlueTrain/main.py \
    --model vit_b \
    --dist-mode bluefog \
    --backend nccl \
    --log-interval 1 \
    --batch-size 64 \
    --epochs 100 \
    --lr 0.0125 \
    --momentum 0.9 \
    --world-size 8 \


# for bluefog