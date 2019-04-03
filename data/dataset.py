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



##-----------用于multi 和　RGB最原始数据的读取，用于MTCNN的检测识别-----##
def load_rgb(imgpath, dsize=None):
    # imgpath　直接是文件的具体名称，/haitao/zxy/1.bmp
    # dsize 是要不要改变尺寸的大小，dsize=(64,64)
    """
    Params:
        imgpath: {str}
        dsize:   {tuple(W, H)}
    Returns:
        imgs:   {ndarray(H, W, 3)}
    """
    assert os.path.exists(imgpath), "rgb file does not exist!"
    img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
    if dsize is not None:
        img = cv2.resize(img, dsize)
    return img

def load_multi(imgdir, dsize=None):
    # imgdir是46个波段的目录，下面的程序，就是要把这46个波段合并成一张图片
    """
    Params:
        imgdir: {str}
        dsize:   {tuple(W, H)}
    Returns:
        imgs:   {ndarray(H, W, C)}
    """
    assert os.path.exists(imgdir), "multi directory does not exist!"
    imgfiles = os.listdir(imgdir)
    
    # 根据波长排序
    wavelength = []
    for imgfile in imgfiles:
        # e.g. imgfile = W550E1000_550.bmp
        wavelength += [int(imgfile.split('.')[0].split('_')[-1])] # 550
    imgfiles = np.array(imgfiles); wavelength = np.array(wavelength)
    imgfiles = imgfiles[np.argsort(wavelength)]
    imgfiles = list(imgfiles)
    
    # 载入图像
    c = len(imgfiles)
    if dsize is None:
        img = cv2.imread(os.path.join(imgdir, imgfiles[0]), cv2.IMREAD_GRAYSCALE)
        (h, w) = img.shape
    else:
        (w, h) = dsize
    imgs = np.zeros(shape=(h, w, c), dtype='uint8')
    for i in range(c):
        img = cv2.imread(os.path.join(imgdir, imgfiles[i]), cv2.IMREAD_GRAYSCALE)
        imgs[:, :, i] = img if (dsize is None) else cv2.resize(img, dsize)
    return imgs

##---------------------用于HSI图片的读取,从npy中读取数据----------------------------##
class HyperSpectralFace(Dataset):
    
    def __init__(self,mode,transforms=None):
        '''
        主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据
        '''
        # 这里不加载实际图片，只是制定路径，当调用__getitem__时才会真正读图片
        if mode == "train":
            f = open("/home/xianyi/new_split/split_1/train.txt", "r")
        elif mode == "valid":
            f = open("/home/xianyi/new_split/split_1/valid.txt", "r")
        else:
            f = open("/home/xianyi/new_split/split_1/test.txt", "r")
        
        # img(f): /home/louishsu/Work/Workspace/ECUST2019/DATA1/1/Multi/non-obtructive/Multi_1_W1_1/W810E620_810.bmp
        # img(win): E:/ECUST2019_NPY_new//DATA1/1/Multi/non-obtructive/Multi_1_W1_1/810.npy
        # img(ubuntu): /media/haitao/zxy/ECUST2019_NPY_new/DATA1/1/Multi/non-obtructive/Multi_1_W1_1/810.npy
        self.imgs = []
        for img in f:
            img = img.split('/',6)[-1].rsplit('/',1)[0]+'/'+img.split('_')[-1].split('.')[0]+'.npy'
            # transform filename for windows
            # img = os.path.join('E:/ECUST2019_NPY_new/', img)
            # transform filename for ubuntu
            img = os.path.join('/media/haitao/zxy/ECUST2019_NPY_new/', img)
            self.imgs.append(img)
        f.close()

       
    def __getitem__(self, index):
        '''
        返回一张图片的数据
        '''
        data = np.load(self.imgs[index])
        data = np.resize(data, (112, 112)).reshape((1, 112, 112))
        data = data.astype(np.float32)/255
        data = torch.Tensor(data)
        # label for windows
        # label = int(self.imgs[index].split("/", 3)[-1].split("/")[0]) - 1   # output 的索引从0开始，所以label也应该从0开始
        # label for ubuntu
        label = int(self.imgs[index].split("/", 6)[-1].split("/")[0]) - 1 
        return data, label
        
    
    def __len__(self):
        return len(self.imgs)

# ---------------------用于HSI图片的读取 从原图中读取----------------------------#
notUsedSubjects = []
get_vol = lambda i: (i-1)//10+1

def getDicts():
    dicts = dict()
    for vol in ["DATA%d" % _ for _ in range(1, 5)]:
        txtfile = os.path.join(opt.data_rootdir, vol, "detect.txt")
        with open(txtfile, 'r') as f:
            dicts[vol] = eval(f.read())
    return dicts    

def get_label_from_path(path):
    path_split = path.split('/')
    idx = path_split.index('ECUST2019')
    label = int(path_split[idx+2])
    return label

class HSI_from_original(Dataset):
    labels = [i for i in range(1, 41) if (i not in notUsedSubjects)]
    def __init__(self,mode,transforms=None):
        '''目标:根据.txt文件获取所有图片的地址 '''
        
        self.imgpath_list=[]
        
        if mode == "train":
            f = open(opt.train_split, "r")
        elif mode == "valid":
            f = open(opt.valid_split, "r")
        else:
            f = open(opt.test_split, "r")
        self.contents=f.readlines()                                                #读取文档中的所有行
        self.facesize = tuple((64,64))
        self.dicts = getDicts()
        
    def __getitem__(self, index):
        filename = self.contents[index].strip()
        filename = os.path.join(opt.data_rootdir, filename)
        label = get_label_from_path(filename)

        # get bbox
        vol = "DATA%d" % get_vol(label)
        imgname = filename[filename.find("DATA")+5:]
        dirname = '/'.join(imgname.split('/')[:-1])
        bbox = self.dicts[vol][dirname][1]
        [x1, y1, x2, y2] = bbox

        # load image array
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)[y1: y2, x1: x2]
        if self.facesize is not None:
            image = cv2.resize(image, self.facesize[::-1])

        image = image[:, :, np.newaxis]
        image = ToTensor()(image)
        label = self.labels.index(label)
        return image, label
    
    def __len__(self):
        return len(self.contents)


# ---------------用于HSI图片的读取，从原图中读取,变成数组，用于传统方法-----------------#
def HSI_readall(mode,transforms=None):
    '''
    主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据
    '''
    # 这里不加载实际图片，只是制定路径，当调用__getitem__时才会真正读图片
    labels = [i for i in range(1, 41) if (i not in notUsedSubjects)]
    imgpath_list=[]

    if mode == "train":
        f = open("/media/haitao/zxy/new_split/split1/train.txt", "r")
    elif mode == "valid":
        f = open("/media/haitao/zxy/new_split/split1/valid.txt", "r")
    else:
        f = open("/media/haitao/zxy/new_split/split1/test.txt", "r")

    contents=f.readlines()                                                #读取文档中的所有行
    facesize = tuple((64,64))
    dicts = getDicts()

    num_sample = len(contents)

    image_list =np.zeros([num_sample,4096])
    label_list = np.zeros(num_sample)

    for i in range(num_sample):
        filename = contents[i].strip()
        filename = os.path.join('/media/haitao/zxy/ECUST2019' ,filename)
        label = get_label_from_path(filename)

        # get bbox
        vol = "DATA%d" % get_vol(label)
        imgname = filename[filename.find("DATA")+5:]
        dirname = '/'.join(imgname.split('/')[:-1])
        bbox = dicts[vol][dirname][1]
        [x1, y1, x2, y2] = bbox

        # load image array
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)[y1: y2, x1: x2]
        if facesize is not None:
            image = cv2.resize(image, facesize[::-1])

        image = image[:, :, np.newaxis].reshape(64*64)

        label = labels.index(label)

        image_list[i] = image
        label_list[i] = label

    return image_list,label_list