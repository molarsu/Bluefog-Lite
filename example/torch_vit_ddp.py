import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.models import vit_b_16, ViT_B_16_Weights
from tqdm import tqdm
import argparse
import logging

def setup(rank, args):
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port

    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=args.world_size)

    # 设置随机种子
    torch.manual_seed(args.seed)

def cleanup():
    dist.destroy_process_group()

def train(rank, args):
    setup(rank, args)

    # 设置日志
    if rank == 0:
        logging.basicConfig(filename=os.path.join(args.output, 'train.log'), level=logging.INFO, format='%(asctime)s - %(message)s')
        logger = logging.getLogger()
        logger.addHandler(logging.StreamHandler())

    # 构建ViT模型
    weights = ViT_B_16_Weights.DEFAULT
    model = vit_b_16(weights=weights)
    model.heads[0] = nn.Linear(model.heads[0].in_features, 100)  # 修改分类头为100类
    model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    # 设置数据集
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
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                              download=True, transform=transform_train)
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, num_replicas=args.world_size, rank=rank)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=2,
                                               sampler=train_sampler)
    testset = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # 训练模型
    for epoch in range(args.epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        total, correct=0, 0
        with tqdm(trainloader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch") as t_bar:
            for i, data in enumerate(t_bar, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(rank), data[1].to(rank)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = ddp_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                _, pred = outputs.max(dim=1)
                total += labels.size(dim=0)
                correct += pred.eq(labels).sum().item()
                t_bar.set_postfix(loss=running_loss / (i + 1), train_acc=correct/total)
        t_bar.close()
        # 每10个epoch进行一次测试
        if (epoch + 1) % 10 == 0 or epoch==0:
            test_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for data in testloader:
                    inputs, labels = data[0].to(rank), data[1].to(rank)
                    outputs = ddp_model(inputs)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            accuracy = 100 * correct / total
            if rank == 0:
                logging.info(f'Epoch {epoch+1}, Test Loss: {test_loss/len(testloader):.3f}, Test Accuracy: {accuracy:.3f}%')

            # 将测试结果写入日志文件
            if rank == 0:
                with open(os.path.join(args.output, 'test.log'), 'a') as f:
                    f.write(f'Epoch {epoch+1}, Test Loss: {test_loss/len(testloader):.3f}, Test Accuracy: {accuracy:.3f}%\n')

        # 将训练过程中的loss写入日志文件
        if rank == 0:
            logging.info(f'Epoch {epoch+1}, Training Loss: {running_loss / len(trainloader):.3f}')

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Distributed Training Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--learning-rate', type=float, default=0.0125, metavar='LR',
                        help='learning rate (default: 0.0125)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--world-size', type=int, default=8, metavar='WS',
                        help='number of workers to train (default: 8)')
    parser.add_argument('--seed', type=int, default=44,
                        help='torch seed to set (default:44)')
    parser.add_argument('--master-addr', type=str, default='localhost',
                        help='master address (default:localhost)')
    parser.add_argument('--master-port', type=str, default='44444',
                        help='master port (default:44444)')
    parser.add_argument('--output', type=str, default='output',
                        help='output directory for logs (default: output)')
    args = parser.parse_args()
    args.output = os.path.join(args.output, f'world_size_{args.world_size}_batch_size_{args.batch_size}_lr_{args.learning_rate}_epochs_{args.epochs}')
    import time
    current_time = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    args.output = os.path.join(args.output, f'{current_time}')
    print(args.output)
    # import sys
    # sys.exit(0)
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    # 启动多进程训练
    mp.spawn(train, args=(args,), nprocs=args.world_size, join=True)
