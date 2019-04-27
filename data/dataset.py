#coding:utf-8
import os
import sys
import cv2
import numpy as np
import torch
from PIL import  Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler
from torchvision import  transforms as T
from torchvision.datasets import MNIST 
from config import opt
import scipy.misc
import imageio
from torchvision.transforms import ToTensor


##-----------
def get_label_from_path(path):
    path_split = path.split('/')
    idx = path_split.index('yperspetral')
    label = int(path_split[idx+1])-4095?
    return label

def HSI_Scenes_dataset_SVM():

    ## ------总共有多少张图片
    rootdir = '' # 改成数据存放的父目录
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    for people in range(4094,4097):?
        content = [] 
        filename = rootdir + '/' +'hyperspetral/{}'.format(people)?
        files= os.listdir(filename) # 得到文件夹下的所有文件名称
        for file in files: #遍历文件夹
            content.append(file) #每个文件的文本存到list中
        ## -------上面得到的content包含了所有的文件-------
        items = len(content)
        train = int(items*0.6)
        test = items - train
        trainfiles = random.sample(content, train)
        testfiles = [f for f in content if f not in trainfiles]
        ## -------上面的trainfiles和testfiles包含了训练和测试的所有文件-------
        for img in trainfiles:
            image = cv2.imread(img, cv2.IMREAD_GRAYSCALE).reshape(224*224*1)
            label = get_label_from_path(img)
            train_data.append(image)
            train_label.append(label)
        for img in testfiles:
            image = cv2.imread(img, cv2.IMREAD_GRAYSCALE).reshape(224*224*1)
            label = get_label_from_path(img)
            test_data.append(image)
            test_label.append(label)

    return train_data,train_label,test_data,test_label

def HSI_Scenes_dataset_NN():

    ## ------总共有多少张图片
    rootdir = '' # 改成数据存放的父目录
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    for people in range(4094,4097):?
        content = [] 
        filename = rootdir + '/' +'hyperspetral/{}'.format(people)?
        files= os.listdir(filename) # 得到文件夹下的所有文件名称
        for file in files: #遍历文件夹
            content.append(file) #每个文件的文本存到list中
        ## -------上面得到的content包含了所有的文件-------
        items = len(content)
        train = int(items*0.6)
        test = items - train
        trainfiles = random.sample(content, train)
        testfiles = [f for f in content if f not in trainfiles]
        ## -------上面的trainfiles和testfiles包含了训练和测试的所有文件-------
        for img in trainfiles:
            image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
            label = get_label_from_path(img)
            train_data.append(image)
            train_label.append(label)
        for img in testfiles:
            image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
            label = get_label_from_path(img)
            test_data.append(image)
            test_label.append(label)

    return train_data,train_label,test_data,test_label