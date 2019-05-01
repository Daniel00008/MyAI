#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 00:17:29 2019
MTCNN_Face_Detecton_Inference
@author: dengdeng
"""

import os
import phpserialize
#import socket
import cv2

import argparse
import os.path

import sys

from PIL import Image, ImageFont, ImageDraw # 导入模块
import os
import time 
import numpy as np
#from six.moves import urllib
import tensorflow as tf
from src.align import detect_face


FLAGS = None

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

_RESIZE_SIDE_MIN = 256
_RESIZE_SIDE_MAX = 512


mtcnn_model_dir = 'src/align'

with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        with tf.variable_scope('pnet'):
            data = tf.placeholder(tf.float32, (None,None,None,3), 'input')
            pnet = detect_face.PNet({'data':data})
            pnet.load(mtcnn_model_dir + '/det1.npy', sess)
        with tf.variable_scope('rnet'):
            data = tf.placeholder(tf.float32, (None,24,24,3), 'input')
            rnet = detect_face.RNet({'data':data})
            rnet.load(mtcnn_model_dir + '/det2.npy', sess)
        with tf.variable_scope('onet'):
            data = tf.placeholder(tf.float32, (None,48,48,3), 'input')
            onet = detect_face.ONet({'data':data})
            onet.load(mtcnn_model_dir + '/det3.npy', sess)

        pnet_fun = lambda img : sess.run(('pnet/conv4-2/BiasAdd:0', 'pnet/prob1:0'), feed_dict={'pnet/input:0':img})
        rnet_fun = lambda img : sess.run(('rnet/conv5-2/conv5-2:0', 'rnet/prob1:0'), feed_dict={'rnet/input:0':img})
        onet_fun = lambda img : sess.run(('onet/conv6-2/conv6-2:0', 'onet/conv6-3/conv6-3:0', 'onet/prob1:0'), feed_dict={'onet/input:0':img})

def get_box(box_ser):
    # a:4:{i:0;a:2:{i:0;i:499;i:1;i:909;}i:1;a:2:{i:0;i:538;i:1;i:909;}i:2;a:2:{i:0;i:538;i:1;i:957;}i:3;a:2:{i:0;i:499;i:1;i:957;}}
    # {0: {0: 499, 1: 909}, 1: {0: 538, 1: 909}, 2: {0: 538, 1: 957}, 3: {0: 499, 1: 957}}
    points = phpserialize.unserialize(box_ser)
    box = [
        points[0][0],
        points[0][1],
        points[2][0],
        points[2][1]
    ]
    return box

def draw_box(img, box, score):
    img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 5)
    # img = cv2.putText(img, str('%3.f' % float(score)), (box[0], box[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
    return img

# 获取人脸的实际坐标, 因为box可能超出了原图的边界,这里需要用0或者最大值代替
def get_real_box(box, img_max_w, img_max_h):

    if box[0] <= 0:
        box[0] = 0
    if box[0] >= img_max_w:
        box[0] = img_max_w

    if box[2] <= 0:
        box[2] = 0
    if box[2] >= img_max_w:
        box[2] = img_max_w

    if box[1] <= 0:
        box[1] = 0
    if box[1] >= img_max_h:
        box[1] = img_max_h

    if box[3] <= 0:
        box[3] = 0
    if box[3] >= img_max_h:
        box[3] = img_max_h

    return box


def detectFace_mtcnn(in_img):

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    
    img = np.asarray(in_img)    
    img=img[...,::-1]
    
    bounding_boxes, points = detect_face.detect_face(img, minsize, pnet_fun, rnet_fun, onet_fun, threshold, factor)
    if len(bounding_boxes) == 0:
        print('no human face')
        return 0

    box = get_real_box(bounding_boxes[0][0:4], img.shape[1], img.shape[0])

    x_left = int(box[0])
    x_right = int(box[2])
    y_up = int(box[1])
    y_down = int(box[3])
        
    return bounding_boxes

#图像保存路径
file_path='./test_imgs/'
#图像名称
pic_name ='Averange4_2.jpg'

pic_name_box='Averange4_2_box.jpg'

img_path=os.path.join(file_path,pic_name)

img_box_path=os.path.join(file_path,pic_name_box)

#若图像存在则绘制框
if os.path.exists(img_path):
    img=cv2.imread(img_path)
    boxes=detectFace_mtcnn(img)
    
    score=0
    for i in range(len(boxes)):
        box = get_real_box(boxes[i][0:4], img.shape[1], img.shape[0])
        box = list(map(lambda x:int(x),box))
        img_with_boxes=draw_box(img,box,score)
    cv2.imwrite(img_box_path, img_with_boxes)

        
        
        
        
        
        

