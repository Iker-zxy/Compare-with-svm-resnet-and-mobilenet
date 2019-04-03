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
import time
import cv2

# 画曲线
from tensorboardX import SummaryWriter
writer = SummaryWriter('log_facenet')

# 导入参数
from config import opt

# 导入模型
import models

# 导入数据
from data.facenet_dataloader import TripletFaceDataset, get_dataloader


# 导入损失函数
from utils.loss import TripletLoss


# 导入评价指标
from utils.metrics import evaluate, plot_roc

# 判断是GPU还是CPU pytorch 0.4 版本之后推荐的写法
device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 用到的函数
l2_dist = PairwiseDistance(2)
from mtcnn_main import mtcnn_simplify

## 定义网络
## 定义数据
## 定义损失函数和优化器
## 计算重要指标
## 开始训练
##    训练网络
##    可视化各种指标
##    计算在验证集上的指标
def train():
    prev_time = datetime.now()

    # -----------------------step1: 模型---------------------------
    model = getattr(models, opt.model)().to(device)
    if opt.load_model_path:
        model.load(opt.load_model_path)

    # -------------------step2: 损失函数和优化器----------------------
    # 特定的TripletLoss损失函数
    optimizer = torch.optim.Adam(model.parameters(),lr = opt.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.step_size,
                            gamma=0.1, last_epoch=-1)
    
    # step4: 统计指标：平滑处理之后的损失，还有混淆矩阵

    # -----------------------step3: 训练----------------------------
    for epoch in range(opt.max_epoch):
        print(80 * '=')
    # -----------------------step４: 数据----------------------------
    # generate triplets online
        train_dataloaders, train_datasize = get_dataloader(opt.train_data_root,
                                            opt.train_csv_name, 
                                            opt.num_train_triplets,
                                            opt.batch_size,
                                            opt.num_workers)
                              
        labels, distances = [], []
        train_loss_sum = 0

        scheduler.step() # 学习率调整
        model.train()

        for batch_idx, batch_sample in enumerate(train_dataloaders):

            anc_img = batch_sample['anc_img'].to(device)
            pos_img = batch_sample['pos_img'].to(device)
            neg_img = batch_sample['neg_img'].to(device)
        
            pos_cls = batch_sample['pos_class'].to(device)
            neg_cls = batch_sample['neg_class'].to(device)

            # forward

            # anc_embed, pos_embed and neg_embed are encoding(embedding) of image
            anc_embed = model(anc_img)
            pos_embed = model(pos_img)
            neg_embed = model(neg_img)
            
            # choose the hard negatives only for "training"
            pos_dist = l2_dist.forward(anc_embed, pos_embed)
            neg_dist = l2_dist.forward(anc_embed, neg_embed)
            allkind = (neg_dist - pos_dist < opt.margin).cpu().numpy().flatten()
            hard_triplets = np.where(allkind == 1)
            if len(hard_triplets[0]) == 0:
                continue  # continue的作用:当没有hard_triplet时,会跳过接下去的程序,直接从上面的for开始新的循环
            
            anc_hard_embed = anc_embed[hard_triplets].to(device)
            pos_hard_embed = pos_embed[hard_triplets].to(device)
            neg_hard_embed = neg_embed[hard_triplets].to(device)
        
            anc_hard_img   = anc_img[hard_triplets].to(device)
            pos_hard_img   = pos_img[hard_triplets].to(device)
            neg_hard_img   = neg_img[hard_triplets].to(device)
        
            pos_hard_cls   = pos_cls[hard_triplets].to(device)
            neg_hard_cls   = neg_cls[hard_triplets].to(device)
        
            anc_img_pred   = model.forward_classifier(anc_hard_img).to(device)
            pos_img_pred   = model.forward_classifier(pos_hard_img).to(device)
            neg_img_pred   = model.forward_classifier(neg_hard_img).to(device)

            loss  = TripletLoss(opt.margin).forward(anc_hard_embed, pos_hard_embed, neg_hard_embed).to(device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            dists = l2_dist.forward(anc_embed, pos_embed)
            distances.append(dists.data.cpu().numpy())
            labels.append(np.ones(dists.size(0))) 

            dists = l2_dist.forward(anc_embed, neg_embed)
            distances.append(dists.data.cpu().numpy())
            labels.append(np.zeros(dists.size(0)))

            # 更新统计指标以及可视化 
            train_loss_sum += loss.item() # 损失
        
        model.save()

        # 统计指标以及可视化 
        avg_triplet_loss = train_loss_sum / train_datasize
        labels           = np.array([sublabel for label in labels for sublabel in label])
        distances        = np.array([subdist for dist in distances for subdist in dist])
        tpr, fpr, accuracy, val, val_std, far = evaluate(distances, labels)
        train_acc = np.mean(accuracy)
        train_loss = avg_triplet_loss
        epoch_str = ("Epoch {}. Train Loss: {:.4f}, Train Acc: {:.4f} ".format(epoch,
                                         train_loss, train_acc))

        # 打印运行时间    
        cur_time = datetime.now()
        h, remainder = divmod((cur_time-prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = ",Time %02d:%02d:%02d" % (h, m, s)

        print(epoch_str + time_str)
        prev_time = cur_time
        

        # 验证
        if epoch % opt.valid_freq == 0 :

            with torch.no_grad(): # operations inside don't track history

                valid_dataloaders, valid_datasize = get_dataloader(opt.valid_data_root,
                                                    opt.valid_csv_name, 
                                                    opt.num_valid_triplets,
                                                    opt.batch_size,
                                                    opt.num_workers)     
                
                labels, distances = [], []
                valid_loss_sum = 0.0

                model.eval()

                for batch_idx, batch_sample in enumerate(valid_dataloaders):

                    anc_img = batch_sample['anc_img'].to(device)
                    pos_img = batch_sample['pos_img'].to(device)
                    neg_img = batch_sample['neg_img'].to(device)
                
                    pos_cls = batch_sample['pos_class'].to(device)
                    neg_cls = batch_sample['neg_class'].to(device)
                    
                    # forward
                    # anc_embed, pos_embed and neg_embed are encoding(embedding) of image
                    anc_embed = model(anc_img)
                    pos_embed = model(pos_img)
                    neg_embed = model(neg_img)
            
                    # choose the hard negatives only for "training"
                    pos_dist = l2_dist.forward(anc_embed, pos_embed)
                    neg_dist = l2_dist.forward(anc_embed, neg_embed)
                    allkind = (neg_dist - pos_dist < opt.margin).cpu().numpy().flatten()
                    
                    hard_triplets = np.where(allkind >= 0)
                    
                    anc_hard_embed = anc_embed[hard_triplets].to(device)
                    pos_hard_embed = pos_embed[hard_triplets].to(device)
                    neg_hard_embed = neg_embed[hard_triplets].to(device)
                
                    anc_hard_img   = anc_img[hard_triplets].to(device)
                    pos_hard_img   = pos_img[hard_triplets].to(device)
                    neg_hard_img   = neg_img[hard_triplets].to(device)
                
                    pos_hard_cls   = pos_cls[hard_triplets].to(device)
                    neg_hard_cls   = neg_cls[hard_triplets].to(device)
                
                    anc_img_pred   = model.forward_classifier(anc_hard_img).to(device)
                    pos_img_pred   = model.forward_classifier(pos_hard_img).to(device)
                    neg_img_pred   = model.forward_classifier(neg_hard_img).to(device)
                
                    loss = TripletLoss(opt.margin).forward(anc_hard_embed, pos_hard_embed, neg_hard_embed).to(device)

                    dists = l2_dist.forward(anc_embed, pos_embed)
                    distances.append(dists.data.cpu().numpy())
                    labels.append(np.ones(dists.size(0))) 

                    dists = l2_dist.forward(anc_embed, neg_embed)
                    distances.append(dists.data.cpu().numpy())
                    labels.append(np.zeros(dists.size(0)))

                    valid_loss_sum += loss.item()

                # 统计指标以及可视化 
                avg_triplet_loss = valid_loss_sum / valid_datasize
                labels           = np.array([sublabel for label in labels for sublabel in label])
                distances        = np.array([subdist for dist in distances for subdist in dist])
                tpr, fpr, accuracy, val, val_std, far = evaluate(distances, labels)
                valid_loss = avg_triplet_loss
                valid_acc = np.mean(accuracy)
                epoch_str = ("Valid Loss: {:.4f}, Valid Acc: {:.4f} ".format(valid_loss, valid_acc))

                # 计算运行时间    
                cur_time = datetime.now()
                h, remainder = divmod((cur_time-prev_time).seconds, 3600)
                m, s = divmod(remainder, 60)
                time_str = ",Time %02d:%02d:%02d" % (h, m, s)

                print(epoch_str + time_str)

                # 画ROC曲线,并且保存
                prefix = './pictures/' + 'facenet_valid' + 'roc_'
                name = time.strftime(prefix + '%m%d_%H%M%S.png')
                plot_roc(fpr, tpr, figure_name = name)

                # 画tensorboard
                writer.add_scalars('Accurary', {'train':train_acc, 'valid':valid_acc},epoch)
                writer.add_scalars('Loss',  {'train':train_loss, 'valid':valid_loss},epoch)

                prev_time = cur_time

def test():
    
    # ----------------------------------模型-------------------------------
    model = getattr(models, opt.model)().to(device)
    if opt.load_model_path:
        model.load(opt.load_model_path)

    # --------------------------------- 数据------------------------------
    test_dataloaders, test_datasize = get_dataloader(opt.test_data_root,
                                        opt.test_csv_name, 
                                        opt.num_test_triplets,
                                        opt.batch_size,
                                        opt.num_workers)

   # ----------------------------------测试－－－－－－－－－－－－－－－－－－
    with torch.no_grad(): # operations inside don't track history

        labels, distances = [], []
        test_loss_sum = 0.0

        model.eval()

        for batch_idx, batch_sample in enumerate(test_dataloaders):

            anc_img = batch_sample['anc_img'].to(device)
            pos_img = batch_sample['pos_img'].to(device)
            neg_img = batch_sample['neg_img'].to(device)
        
            pos_cls = batch_sample['pos_class'].to(device)
            neg_cls = batch_sample['neg_class'].to(device)
            
            # forward
            # anc_embed, pos_embed and neg_embed are encoding(embedding) of image
            anc_embed = model(anc_img)
            pos_embed = model(pos_img)
            neg_embed = model(neg_img)
    
            # choose the hard negatives only for "training"
            pos_dist = l2_dist.forward(anc_embed, pos_embed)
            neg_dist = l2_dist.forward(anc_embed, neg_embed)
            allkind = (neg_dist - pos_dist < opt.margin).cpu().numpy().flatten()
            
            hard_triplets = np.where(allkind >= 0)
            
            anc_hard_embed = anc_embed[hard_triplets].to(device)
            pos_hard_embed = pos_embed[hard_triplets].to(device)
            neg_hard_embed = neg_embed[hard_triplets].to(device)
        
            anc_hard_img   = anc_img[hard_triplets].to(device)
            pos_hard_img   = pos_img[hard_triplets].to(device)
            neg_hard_img   = neg_img[hard_triplets].to(device)
        
            pos_hard_cls   = pos_cls[hard_triplets].to(device)
            neg_hard_cls   = neg_cls[hard_triplets].to(device)
        
            anc_img_pred   = model.forward_classifier(anc_hard_img).to(device)
            pos_img_pred   = model.forward_classifier(pos_hard_img).to(device)
            neg_img_pred   = model.forward_classifier(neg_hard_img).to(device)
        
            loss = TripletLoss(opt.margin).forward(anc_hard_embed, pos_hard_embed, neg_hard_embed).to(device)

            dists = l2_dist.forward(anc_embed, pos_embed)
            distances.append(dists.data.cpu().numpy())
            labels.append(np.ones(dists.size(0))) 

            dists = l2_dist.forward(anc_embed, neg_embed)
            distances.append(dists.data.cpu().numpy())
            labels.append(np.zeros(dists.size(0)))

            test_loss_sum += loss.item()

        # 统计指标以及可视化 
        avg_triplet_loss = test_loss_sum / test_datasize
        labels           = np.array([sublabel for label in labels for sublabel in label])
        distances        = np.array([subdist for dist in distances for subdist in dist])
        tpr, fpr, accuracy, val, val_std, far = evaluate(distances, labels)

        epoch_str = ("test Loss: {:.4f}, Train Acc: {:.4f} ".format(avg_triplet_loss, 
                                                                    np.mean(accuracy)))

        print(epoch_str)

        # 画ROC曲线,并且保存
        prefix = './pictures/' + 'facenet_test_' + 'roc_'
        name = time.strftime(prefix + '%m%d_%H%M%S.png')
        plot_roc(fpr, tpr, figure_name = name)

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

def img_to_embedding(image_path, model):

    img = mtcnn_simplify(image_path,182,44,1,False)
    # size = opt.netinputsize
    # img = cv2.resize(img,(size,size))
    img = np.transpose(img,(2,0,1))
    img = np.array([img])
    img = torch.Tensor(img).to(device)

    embedding = model(img).cpu().detach().numpy()
    return embedding

def verify(image_path, identity):
    """
    Function that verifies if the person on the "image_path" image is "identity".

    Arguments:
    image_path -- path to an image
    identity -- string, name of the person you'd like to verify the identity. Has to be a resident of the Happy house.
    database -- python dictionary mapping names of allowed people's names (strings) to their encodings (vectors).
    model -- your Inception model instance in Keras

    Returns:
    dist -- distance between the image_path and the image of "identity" in the database.
    door_open -- True, if the door should open. False otherwise.
    """
    # ------------------------import model-----------------------
    model = getattr(models, opt.model)().to(device)
    if opt.load_model_path:
        model.load(opt.load_model_path)
    model.eval()

    # ---------------------建立需要识别的人脸数据集------------------
    database = {}
    database["cyr"] = img_to_embedding("pictures/4_1.JPG",model)

    # Step 1: Compute the encoding for the image. Use img_to_embedding() see example above. (≈ 1 line)
    embedding = img_to_embedding(image_path,model)

    # Step 2: Compute distance with identity's image (≈ 1 line)
    dist = np.linalg.norm(embedding  - database[identity])

    # Step 3: Open the door if dist < 0.7, else don't open (≈ 3 lines)
    if dist < 0.7:
        print("It's " + str(identity))
    else:
        print("It's not " + str(identity))
    print(dist)
    ### END CODE HERE ###

    return dist

if __name__=='__main__':
    device  = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # train()
    # test()
    verify("pictures/5_5.JPG","cyr")