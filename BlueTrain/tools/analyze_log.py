import matplotlib.pyplot as plt
import re
import os
import debugpy
debugpy.connect(('172.16.1.249',5678))
class AnalyzeLog:
    def __init__(self, log_path):
        self.log_path = log_path
        # 2024-04-09 22:01:31,338 - INFO - Epoch 100 Iter 24, Training Loss: 0.002, Training ACC:  1.000
        
        self.pattern_train = r".*?Epoch (\d+) Iter (\d+), Training Loss: ([\d.]+), Training ACC:\s*([\d.]+).*"
        self.pattern_test = r".*?Epoch (\d+), Test Loss: ([\d.]+), Test Accuracy: ([\d.]+)%?.*"

        # 初始化属性
        self.train_loss = []
        self.train_acc = []
        self.test_loss = []
        self.test_acc = []
        self.train_epochs = []
        self.test_epochs = []
    
    def get_log(self):
        with open(self.log_path, 'r') as file:
            for line in file:
                if 'INFO' not in line:
                    continue
                match_train = re.search(self.pattern_train, line)
                match_test = re.search(self.pattern_test, line)
        
                if match_train:
                    train_epoch, _, loss, acc = match_train.groups()
                    self.train_epochs.append(int(train_epoch))
                    self.train_loss.append(float(loss)) 
                    self.train_acc.append(float(acc) * 100)
                    
                elif match_test:
                    test_epoch, loss, acc = match_test.groups()
                    self.test_epochs.append(int(test_epoch))
                    self.test_loss.append(float(loss)) 
                    self.test_acc.append(float(acc))

    def align_train_test(self):
        train_index, test_index = 0, 0
        aligned_train_loss, aligned_test_loss, aligned_train_acc, aligned_test_acc = [], [], [], []
        
        for train_index in range(len(self.train_loss)):
            aligned_train_loss.append(self.train_loss[train_index])
            aligned_train_acc.append(self.train_acc[train_index])
            
            if test_index + 1 < len(self.test_loss) and self.train_epochs[train_index] == self.test_epochs[test_index + 1]:
                test_index += 1
            
            aligned_test_loss.append(self.test_loss[test_index])
            aligned_test_acc.append(self.test_acc[test_index])
        
        self.aligned_train_loss = aligned_train_loss
        self.aligned_test_loss = aligned_test_loss
        self.aligned_train_acc = aligned_train_acc
        self.aligned_test_acc = aligned_test_acc

    def plot(self):
        # 假设训练周期以训练数据为准
        epochs = range(1, len(self.aligned_train_loss) + 1)
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.aligned_train_loss, label='Train Loss')
        plt.plot(epochs, self.aligned_test_loss, label='Test Loss')
        plt.title('Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        fig_name = f"loss_plots.png"
        fig_path = os.path.join(os.path.dirname(self.log_path), fig_name)
        plt.savefig(fig_path)
        print(f"loss plot saved to: {fig_path}")
        plt.close() 

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.aligned_train_acc, label='Train ACC')
        plt.plot(epochs, self.aligned_test_acc, label='Test ACC')
        plt.title('Accuracy Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        fig_name = f"acc_plots.png"
        fig_path = os.path.join(os.path.dirname(self.log_path), fig_name)
        plt.savefig(fig_path)
        print(f"acc plot saved to: {fig_path}")
        plt.close() 

# 示例使用
analyzer = AnalyzeLog(log_path='./outputs/pytorch/vit_b_8gpu/20240409_202618/train.log')
analyzer.get_log()
analyzer.align_train_test()
analyzer.plot()
