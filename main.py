#coding:utf-8
import os
import torch
from torch import nn
from datetime import datetime
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler # 学习率
from torch.nn.modules.distance import PairwiseDistance
import torchvision
from torchvision import transforms
import numpy as np

# 导入参数
from config import opt

# 导入模型
import models
# 导入数据
from data.dataset import HSI_Scenes_dataset_NN

# 导入评价指标
from utils.function import get_acc

def train():
    
    # vis = Visualizer(opt.env)
    prev_time = datetime.now()

    # -----------------------step1: 模型---------------------------
    model = getattr(models, opt.model)()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu: model.cuda()

    # -----------------------step2: 数据----------------------------
    train_data,train_label,test_data,test_label = HSI_Scenes_dataset_NN()

    # -------------------step3: 目标函数和优化器----------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr = opt.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.step_size,
                            gamma=0.1, last_epoch=-1)
    
    # step4: 统计指标：平滑处理之后的损失，还有混淆矩阵

    # 训练
    for epoch in range(opt.max_epoch):
        scheduler.step() # 学习率调整

        train_loss = 0
        train_acc = 0
        model.train()

        # 训练模型参数
        if opt.use_gpu:
            train_data = train_data.cuda()
            train_label = train_label.cuda()

        # forward
        output = model(train_data)
        loss= criterion(output, label)
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 更新统计指标以及可视化 
        train_loss += loss.item() # 损失
        
        # 统计指标
        train_acc += get_acc(output, label)
        
        model.save()

        if epoch % opt.valid_freq == 0 :
            # 计算验证集上的指标以及可视化
            valid_loss = 0
            valid_acc = 0
            model.eval()

            # 训练模型参数
            with torch.no_grad():
                if opt.use_gpu:
                    data = data.cuda()
                    label = label.cuda()
            
            # forward
            output = model(data)
            loss= criterion(output, label)
            
            valid_loss += loss.item()

            valid_acc += get_acc(output, label)

            epoch_str = (
                "Epoch %d. Train Loss: %f, Train Acc: %f, Test Loss: %f, Test Acc: %f "
                % (epoch, train_loss / len(train_data),train_acc / len(train_data),
                    test_loss/ len(test_data),test_acc/ len(test_data),
                    ))

            # 打印运行时间    
            cur_time = datetime.now()
            h, remainder = divmod((cur_time-prev_time).seconds, 3600)
            m, s = divmod(remainder, 60)
            time_str = "Time %02d:%02d:%02d" % (h, m, s)
            print(epoch_str + time_str)
            prev_time = cur_time

if __name__=='__main__':
    train()