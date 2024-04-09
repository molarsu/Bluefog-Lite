# build.py
import torch
import bluefoglite.torch_api as bfl
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights, ViT_L_16_Weights,vit_l_16

from model import *


def build_data_loader(args, mode='Train', rank=None):
    transform_train = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    
    if args.dataset == "CIFAR100":
        if mode == "Train":
            trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                                    download=True, transform=transform_train)
            train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, num_replicas=args.world_size, 
                                                                            rank=rank, seed=args.seed, shuffle=True)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=2,
                                                    sampler=train_sampler)
            return trainloader, train_sampler
        
        elif mode == "Test":
            testset = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)
            testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
            return testloader
    else:
        raise NotImplementedError(f"Dataset{args.dataset} not implemented")


def build_model(args):
    model_name = args.model
    # vit base tiny
    if model_name == "vit_b":
        weights = ViT_B_16_Weights.DEFAULT
        model = vit_b_16(weights=weights)
        model.heads[0] = nn.Linear(model.heads[0].in_features, 100)
    elif model_name == "vit_l":
        weights = ViT_L_16_Weights.DEFAULT
        model = vit_l_16(weights=weights)
        model.heads[0] = nn.Linear(model.heads[0].in_features, 100)
    elif model_name == "vit_tiny":
        model = ViT()
    # resnet 20 32 44 56
    elif model_name == "resnet20":
        model = ResNet20()
    elif model_name == "resnet32":
        model = ResNet32()
    elif model_name == "resnet44":
        model = ResNet44()
    elif model_name == "resnet56":
        model = ResNet56()
    # mlp 
    elif model_name == "MLP":
        model = MLP()
    else:
        raise NotImplementedError("model not implemented")
    return model


def build_dist_optimizer(model, args):
    lr = args.lr * args.world_size
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=args.momentum, weight_decay=5e-5
    )
    if args.dist_mode=="pytorch":
        return optimizer
    base_dist_optimizer = bfl.DistributedAdaptWithCombineOptimizer
    if args.dist_optimizer == "allreduce":
        optimizer = base_dist_optimizer(
            optimizer, model=model, communication_type=bfl.CommunicationType.allreduce
        )
    elif args.dist_optimizer == "neighbor_allreduce":
        optimizer = base_dist_optimizer(
            optimizer,
            model=model,
            communication_type=bfl.CommunicationType.neighbor_allreduce,
        )
    elif args.dist_optimizer == "gradient_allreduce":
        optimizer = bfl.DistributedGradientAllreduceOptimizer(optimizer, model=model)
    else :
        raise NotImplementedError("optimizer not implemented")
    return optimizer