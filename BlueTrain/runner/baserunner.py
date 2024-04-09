import torch
import torch.nn.functional as F
import logging
import os
from tqdm import tqdm
from utils import dynamic_topology_update
class BaseRunner:
    def __init__(self, rank, model, optimizer, args, train_data_loader, train_sampler, test_data_loader, neighbor_gen=None):
        self.rank = rank
        self.model = model
        self.optimizer = optimizer
        self.args = args
        self.train_data_loader = train_data_loader
        self.train_sampler = train_sampler
        self.test_data_loader = test_data_loader
        self.logger = None
        self.neighbor_gen = neighbor_gen

        # 设置日志
        if self.rank == 0:
            print(f'=> Train is to begin, Log is in {args.output_path}')
            os.makedirs(args.output_path, exist_ok=True)
            logging.basicConfig(filename=os.path.join(args.output_path, 'train.log'), level=logging.INFO, format='%(asctime)s - %(message)s')
            self.logger = logging.getLogger()
            file_handler = logging.FileHandler(os.path.join(args.output_path, 'train.log'))
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            # 将文件处理程序添加到日志记录器
            self.logger.addHandler(file_handler)

    def train(self):
        for epoch in range(self.args.epochs):
            self.train_one_epoch(epoch)
            if (epoch + 1) % self.args.log_interval == 0:
                self.test(epoch)

    def train_one_epoch(self, epoch):
        self.model.train()
        train_loss, correct, total = 0, 0, 0
        self.train_sampler.set_epoch(epoch)
        with tqdm(self.train_data_loader, desc=f"Epoch {epoch+1}/{self.args.epochs}", unit="batch") as t_bar:
            for i, data in enumerate(t_bar, 0):
                inputs, labels = data[0].to(self.rank), data[1].to(self.rank)
                if not self.args.disable_dynamic_topology and self.args.dist_mode=='bluefog':
                    dynamic_topology_update(self.neighbor_gen, self.args, self.optimizer)
                if self.args.cuda:
                    inputs, labels = inputs.cuda().to(self.rank), labels.cuda().to(self.rank)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                _, pred = outputs.max(dim=1)
                total += labels.size(dim=0)
                correct += pred.eq(labels).sum().item()
                t_bar.set_postfix(loss=train_loss / (i + 1), train_acc=correct/total)
                if self.rank == 0:
                    self.logger.info(f'Epoch {epoch+1} Iter {i + 1}, Training Loss: {train_loss / (i + 1):.3f}')

    def test(self, epoch):
        if self.rank != 0:
            return
        
        self.model.eval()
        test_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for data in self.test_data_loader:
                inputs, labels = data[0].to(self.rank), data[1].to(self.rank)
                outputs = self.model(inputs)
                loss = F.cross_entropy(outputs, labels)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        accuracy = 100 * correct / total
        self.logger.info(f'Epoch {epoch+1}, Test Loss: {test_loss/len(self.test_data_loader):.3f}, Test Accuracy: {accuracy:.3f}%')