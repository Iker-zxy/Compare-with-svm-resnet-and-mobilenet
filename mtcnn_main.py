"""Performs face alignment and stores face thumbnails in the output directory."""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
from utils import function
from utils import detect_face
import random
from time import sleep
import cv2

def mtcnn_simplify(image_path,image_size,margin,gpu_memory_fraction,detect_multiple_faces):
   
    sleep(random.random())
   
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
    
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    try:
        img = misc.imread(image_path)
    except (IOError, ValueError, IndexError) as e:
        errorMessage = '{}: {}'.format(image_path, e)
        print(errorMessage)
    else:
        if img.ndim<2:
            print('Unable to align "%s"' % image_path)
            os._exit()
        if img.ndim == 2:
            img = function.to_rgb(img)
        img = img[:,:,0:3]

        bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        nrof_faces = bounding_boxes.shape[0]
        # --------需要修改，建立一个文件夹用来储存无法剪裁的文件
        with open('E:/Desktop/unableface.txt','w') as text:
            if nrof_faces>0:
                det = bounding_boxes[:,0:4]
                det_arr = []
                img_size = np.asarray(img.shape)[0:2]

                det_arr.append(np.squeeze(det))

                for i, det in enumerate(det_arr):
                    det = np.squeeze(det)
                    bb = np.zeros(4, dtype=np.int32)
                    bb[0] = np.maximum(det[0]-margin/2, 0)
                    bb[1] = np.maximum(det[1]-margin/2, 0)
                    bb[2] = np.minimum(det[2]+margin/2, img_size[1])
                    bb[3] = np.minimum(det[3]+margin/2, img_size[0])
                    cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                    # scaled 就是检测对齐剪裁过后的结果
                    img_mtcnn = misc.imresize(cropped, (image_size, image_size), interp='bilinear') 
                    # ---------需要修改，为储存的目录
                    image_new_name = 'E:'+image_path.split(':')[-1].split('.')[0]+'.jpg'
                    image_new_path = os.path.dirname(image_new_name)
                    mkdir(image_new_path)
                    cv2.imwrite(image_new_name,img_mtcnn)
                    # print(image_new_name)
            else:
                print('unable to align"%s"'% image_path)
                text.write('%s\n'%(image_path))  

# 读取文件
def list_all_files(rootdir): # used for rgb and npytomat
    _files = []
    filelist = os.listdir(rootdir) #列出文件夹下所有的目录与文件
    for i in range(0,len(filelist)):
        path = os.path.join(rootdir,filelist[i])
        if os.path.isdir(path): # 判断是不是目录
            _files.extend(list_all_files(path))
        if os.path.isfile(path):  #判断是不是文件
            if path.split('.')[-1] == 'bmp':
                if int(path.split('.')[-2][-3:])>=550 or int(path.split('.')[-2][-3:])<=100:
                    _files.append(path)
    return _files

# 建文件
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:                   # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)           # makedirs 创建文件时如果路径不存在会创建这个路径

def main(rootdir):
    image_path = list_all_files(rootdir)
    print('Total number of the picture:%d' % len(image_path))
    for image in image_path:
        # --------------需要修改，只要修改224即可，为剪裁下来的尺寸
        mtcnn_simplify(image,224,44,1,False)


if __name__ == '__main__':
    # ------------需要修改，数据集的原始地址
    main('D:/Desktop/hyperspectral')

