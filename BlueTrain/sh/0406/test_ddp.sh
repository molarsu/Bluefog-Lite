CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=8 \
    --master_port=8888 \
    BlueTrain/main.py \
    --model vit_b \
    --dist-mode pytorch \
    --backend nccl \
    --log-interval 1 \
    --batch-size 64 \
    --epochs 100 \
    --lr 0.0125 \
    --momentum 0.9 \
    --world-size 8 \
    --log-interval 3

