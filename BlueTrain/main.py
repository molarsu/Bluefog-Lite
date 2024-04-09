import time
import debugpy
# debugpy.connect(('172.16.1.249',5678))

import os
import argparse
import logging
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import bluefoglite.torch_api as bfl
from bluefoglite.common import topology

from BlueTrain.utils.build import build_model, build_dist_optimizer, build_data_loader
from bluefoglite.utility import (
    broadcast_parameters,
    broadcast_optimizer_state,
)
from runner.baserunner import BaseRunner



def bluefog_init_and_set(args):
    # Initialize topology
    bfl.init(backend=args.backend)
    topo = topology.RingGraph(bfl.size())
    bfl.set_topology(topo)
    dynamic_neighbor_allreduce_gen = None
    if not args.disable_dynamic_topology:
        dynamic_neighbor_allreduce_gen = topology.GetDynamicOnePeerSendRecvRanks(
            bfl.load_topology(), bfl.rank()
        )
    
    if args.cuda:
        print("using cuda.")
        device_id = bfl.rank() % torch.cuda.device_count()
        torch.cuda.set_device(device_id)
        device = torch.tensor([0.0]).cuda().device
    else:
        print("using cpu")
        device = "cpu"
            # Broadcast parameters & optimizer state
    
    return device, dynamic_neighbor_allreduce_gen


def get_args():
    parser = argparse.ArgumentParser(
        description="Bluefog-Lite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="vit_b", help="model to use")
    parser.add_argument("--output-path", type=str, default="./outputs/Debug", help="output path")
    
    parser.add_argument("--dataset", type=str, default="CIFAR100", help="dataset to use")
    parser.add_argument(
        "--batch-size", type=int, default=64, help="input batch size for training"
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=64, help="input batch size for testing"
    )
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.0125, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument("--dist-mode", type=str, default="bluefog", help="distributed mode")
    parser.add_argument("--local-rank", type=int, default=0, help="local rank")
    parser.add_argument(
        "--dist-optimizer",
        type=str,
        default="neighbor_allreduce",
        help="The type of distributed optimizer. Supporting options are [neighbor_allreduce, allreduce]",
        choices=["neighbor_allreduce", "allreduce","gradient_allreduce"],
    )
    parser.add_argument(
        "--communicate-state-dict",
        action="store_true",
        default=False,
        help="If True, communicate state dictionary of model instead of named parameters",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", default=False, help="disables CUDA training"
    )

    parser.add_argument(
        "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="nccl",
        choices=["gloo", "nccl"],
    )
    parser.add_argument(
        "--profiling",
        type=str,
        default="no_profiling",
        metavar="S",
        help="enable which profiling? default: no",
        choices=["no_profiling", "c_profiling", "torch_profiling"],
    )
    parser.add_argument(
        "--disable-dynamic-topology",
        action="store_true",
        default=True,
        help="Disable each iteration to transmit one neighbor per iteration dynamically.",
    )
    parser.add_argument(
        '--world-size', 
        type=int, 
        default=8, 
        metavar='WS',
        help='number of workers to train (default: 8)'
    )


    args = parser.parse_args()
    # get current_time
    current_time = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    args.output_path = os.path.join(args.output_path, f'{current_time}')
    

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args


def main():
    # args
    args = get_args()
    # seed
    torch.manual_seed(args.seed)

    # build model
    model = build_model(args)

    device = None
    neighbor_gen = None
    # init dist
    if args.dist_mode == "pytorch":
        rank=args.local_rank
        dist.init_process_group(args.backend, rank=rank, world_size=args.world_size)
        rank2 = torch.distributed.get_rank()
        assert(rank==rank2)
        model.to(rank)
        model = DDP(model)
    elif args.dist_mode == "bluefog":
        device, neighbor_gen = bluefog_init_and_set(args)
        rank = bfl.rank()
        model.to(rank)

    
    
    # config optimizer
    optimizer = build_dist_optimizer(model, args)


    if args.dist_mode == "bluefog":   
        broadcast_parameters(model.state_dict(), root_rank=0)
        broadcast_optimizer_state(
            optimizer, root_rank=0, device=next(model.parameters()).device
        )
    
    # config dataset
    train_data_loader, train_sampler =  build_data_loader(args, mode='Train', rank=rank)
    test_data_loader =  build_data_loader(args, mode='Test')
    
    # train and test
    runner = BaseRunner(rank=rank, model=model, optimizer=optimizer, args=args, 
          train_data_loader=train_data_loader, train_sampler=train_sampler, 
          test_data_loader=test_data_loader, neighbor_gen=neighbor_gen)
    runner.train()


if __name__ == "__main__":
    main()
