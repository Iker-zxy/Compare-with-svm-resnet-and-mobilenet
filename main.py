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
from utils.function import ArcMarginProduct
net = ArcMarginProduct(128, 33)
net= net.cuda()
# 导入数据
from data.dataset import HyperSpectralFace
from data.dataset import Siamese
from data.facenet_dataloader import TripletFaceDataset, get_dataloader
# from data.dataset import DogCat
# from utils.visualize import Visualizer

# 导入损失函数
from utils.loss import TripletLoss


# 导入评价指标
from utils.function import get_acc
# from utils.metrics import AccumulatedAccuracyMetric

# from torchnet import meter

# from tqdm import tqdm


## 定义网络
## 定义数据
## 定义损失函数和优化器
## 计算重要指标
## 开始训练
##    训练网络
##    可视化各种指标
##    计算在验证集上的指标
def train():
    
    # vis = Visualizer(opt.env)
    prev_time = datetime.now()

    # -----------------------step1: 模型---------------------------
    model = getattr(models, opt.model)()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu: model.cuda()

    # -----------------------step2: 数据----------------------------
    train_data = Siamese("train")
    train_dataloader  = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True)
    valid_data = Siamese("valid")
    valid_dataloader  = DataLoader(valid_data, batch_size=opt.batch_size, shuffle=True)

    # train_data = HyperSpectralFace("train")
    # train_dataloader  = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True)
    # valid_data = HyperSpectralFace("valid")
    # valid_dataloader  = DataLoader(valid_data, batch_size=opt.batch_size, shuffle=True)

    # data_loaders, data_size = get_dataloader(opt.train_data_root,     opt.valid_data_root,
    #                                         opt.train_csv_name,     opt.valid_csv_name,
    #                                         opt.num_train_triplets, opt.num_valid_triplets,   
    #                                         opt.batch_size,         opt.num_workers)

    # -------------------step3: 目标函数和优化器----------------------
    # criterion = ContrastiveLoss(margin =1)
    criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(),lr = opt.lr)
    # optimizer = torch.optim.SGD(model.parameters(),lr = opt.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.step_size,
                            gamma=0.1, last_epoch=-1)
    
    # step4: 统计指标：平滑处理之后的损失，还有混淆矩阵
    train_loss = 0
    train_acc = 0

    # 训练
    for epoch in range(opt.max_epoch):
        scheduler.step() # 学习率调整

        train_loss = 0
        train_acc = 0
        model.train()
        net.train() # mobilefacenet 专用
        for batch_idx, (data,label) in enumerate(train_dataloader):

            #########################
            ###以下是一般图像专用#######
            # 训练模型参数
            if opt.use_gpu:
                data = data.cuda()
                label = label.cuda()
            
            # forward
            feature = model(data)
            output = net(feature,label) # mobilefacenet专用
            loss= criterion(output, label)
            #####以上是一般图像专用######
            ##########################




            #########################
            ###以下是tripletloss专用###
            # # 训练模型参数
            # label = label if len(label) > 0 else None # 在triplet中的label是none
            # if not type(data) in (tuple, list):
            #     data = (data,)

            # if opt.use_gpu:
            #     data = tuple(d.cuda() for d in data) # 在triplet中的data存在3张图片,需要分别cuda
            #     label = label.cuda()
            
            # # forward
            # output = model(*data)  # 打*的原因是data中存在多张图片

            # if not type(output) in (tuple, list):
            #     output = (output,)

            # if label is not None:
            #     label = (label,)
            #     loss_input = output + label
            # loss_output = criterion(*loss_input)
            # loss = loss_output[0] if type(loss_output) in (tuple, list) else loss_output
            # #####以上是triplet专用######
            ##########################

            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 更新统计指标以及可视化 
            train_loss += loss.item() # 损失
            
            # 统计指标
            # metrics = AccumulatedAccuracyMetric()
            train_acc += get_acc(output, label)
        
        model.save()

        # 打印运行时间    
        # cur_time = datetime.now()
        # h, remainder = divmod((cur_time-prev_time).seconds, 3600)
        # m, s = divmod(remainder, 60)
        # time_str = "Time %02d:%02d:%02d" % (h, m, s)

        if epoch % opt.valid_freq == 0 :
            # 计算验证集上的指标以及可视化
            valid_loss = 0
            valid_acc = 0
            model.eval()
            net.eval()  # mobilefacenet专用
            for batch_idx, (data,label) in enumerate(valid_dataloader):
                #------------------------
                ###以下是一般图像专用#######
                # 训练模型参数
                with torch.no_grad():
                    if opt.use_gpu:
                        data = data.cuda()
                        label = label.cuda()
                
                # forward
                output = model(data)  # 打*的原因是data中存在多张图片
                output_1= net(output,label) # mobilefacenet专用
                loss= criterion(output_1, label)
                #####以上是一般图像专用######
                #------------------------


                #------------------------
                ###以下是tripletloss专用###
                # label = label if len(label) > 0 else None # 在triplet中的label是none
                # if not type(data) in (tuple, list):
                #     data = (data,)


                # with torch.no_grad():
                #     if opt.use_gpu:
                #         data = tuple(d.cuda() for d in data)
                #         label = label.cuda()
            
                # output = model(*data)
                # if not type(output) in (tuple, list):
                #     output = (output,)

                # if label is not None:
                #     label = (label,)
                #     loss_input = output + label
                # loss_output = criterion(*loss_input)
                # loss = loss_output[0] if type(loss_output) in (tuple, list) else loss_output
                #####以上是triplet专用######
                #------------------------

                valid_loss += loss.item()

                valid_acc += get_acc(output, label)


            epoch_str = (
                "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f "
                % (epoch, train_loss / len(train_dataloader),train_acc / len(train_dataloader),
                    valid_loss/ len(valid_dataloader),valid_acc/ len(valid_dataloader),
                    ))

            # epoch_str = (
            #     "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, "
            #     % (epoch, train_loss / len(train_dataloader),
            #         train_acc / len(train_dataloader), valid_loss / len(valid_dataloader),
            #         valid_acc / len(valid_dataloader)))
            
            # # update learning rate
            # if loss_meter.value()[0] > previous_loss:          
            #     lr = lr * opt.lr_decay
            #     # 第二种降低学习率的方法:不会有moment等信息的丢失
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = l

            # 打印运行时间    
            cur_time = datetime.now()
            h, remainder = divmod((cur_time-prev_time).seconds, 3600)
            m, s = divmod(remainder, 60)
            time_str = "Time %02d:%02d:%02d" % (h, m, s)
            print(epoch_str + time_str)
            prev_time = cur_time

def test():
    
    # 模型
    model = getattr(models, opt.model)().eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu: model.cuda()

    # 数据
    test_data = HyperSpectralFace('test')
    test_dataloader  = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False)
    results = []
    for batch_idx,(data,path) in enumerate(test_dataloader):
        with torch.no_grad():
            data = Variable(data)
            if opt.use_gpu: data = data.cuda()
        output = model(data)
        probability = torch.nn.functional.softmax(output)[:,0].data.tolist()
        # label = score.max(dim = 1)[1].data.tolist()
        
        batch_results = [(path_,probability_) for path_,probability_ in zip(path,probability) ]

        results += batch_results
    write_csv(results,opt.result_file)

    return results

def write_csv(results,file_name):
    import csv
    with open(file_name,'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id','label'])
        writer.writerows(results)

def help():
    '''
    打印帮助的信息： python file.py help
    '''
    
    print('''
    usage : python file.py <function> [--args=value]
    <function> := train | test | help
    example: 
            python {0} train --env='env0701' --lr=0.01
            python {0} test --dataset='path/to/dataset/root/'
            python {0} help
    avaiable args:'''.format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)

if __name__=='__main__':
    train()