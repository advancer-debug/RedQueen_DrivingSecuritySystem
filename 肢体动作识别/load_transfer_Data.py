#!/usr/bin/env python3
# coding=utf-8

'''
将数据集转变为成tfrecords形式
便于动态读取
'''


_author_ = 'zixuwang'
_datetime_ = '2018-3-25'



import tensorflow as tf
tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES
from PIL import Image
import os

TRAINING_DATA = './trainSet'
VALIDATION_IMAGE = './valiSet'
TEST_IMAGE = './imgs/test'
# 图片尺寸：800*600 RGB
IMAGE_WIDTH = 640
IMAGE_HIGHT = 480
COLOR_CHANNELS = 3

import random

# 使用tf.train.Example来定义我们要填入的数据格式，然后使用tf.python_io.TFRecordWriter来写入
def createData(path,type = 'train'):
    if type == 'train':
        fileName = 'TureTrainSet.tfrecords'
    elif type == 'test':
        fileName = 'TureTestSet.tfrecords'
    else:
        fileName = 'TureValidationSet.tfrecords'

    writer = tf.python_io.TFRecordWriter(fileName)
    imgList = os.listdir(path)
    random.shuffle(imgList)
    for file in imgList:
        img_path = path + '/' + file
        label = int(file.split('#')[0])
        print(img_path,label)
        img = Image.open(img_path)
        img = img.resize((IMAGE_WIDTH,IMAGE_HIGHT))
        img_raw = img.tobytes()  # 将图片转化为原生bytes
        example = tf.train.Example(features=tf.train.Features(feature={
        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
           }))
        # 一个Example中包含Features，Features里包含Feature（这里没s）的字典。最后，Feature里包含有一个 FloatList， 或者ByteList，或者Int64List
        writer.write(example.SerializeToString())
    writer.close()
    # 就这样，我们把相关的信息都存到了一个文件中，而且不用单独的label文件。读取也很方便。



# createData(TRAINING_DATA)
createData(VALIDATION_IMAGE,type='vali')