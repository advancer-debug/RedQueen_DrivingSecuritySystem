#!/usr/bin/env python3
# coding=utf-8

import tensorflow as tf
import math
import time
from PIL import Image
import numpy as np
import os
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt

def createData(path):
    writer = tf.python_io.TFRecordWriter("Training.tfrecords")
    num_classes = os.listdir(path)  # 列出当前目录下的子目录
    for className in num_classes:
        class_path = path + '/' + className
        print(class_path)
        for img_name in os.listdir(class_path):
            print(img_name)
            img_path = class_path + '/' + img_name
            img = Image.open(img_path)
            img = img.resize((480, 640))
            img_raw = img.tobytes()  # 将图片转化为原生bytes
            example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(className)])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                }))
            # 一个Example中包含Features，Features里包含Feature（这里没s）的字典。最后，Feature里包含有一个 FloatList， 或者ByteList，或者Int64List
            writer.write(example.SerializeToString())
    writer.close()
#     就这样，我们把相关的信息都存到了一个文件中，而且不用单独的label文件。读取也很方便。


'''
定义读取数据集的函数
'''
def loadData(path):
    # createData(path)
    # 创建文件队列,不限读取的数量
    filename_queue = tf.train.string_input_producer(["TrainingSet.tfrecords"])
    # create a reader from file queue
    reader = tf.TFRecordReader()
    # reader从文件队列中读入一个序列化的样本
    _, serialized_example = reader.read(filename_queue)
    # get feature from serialized example
    # 解析符号化的样本
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'features': tf.FixedLenFeature([48*48],tf.int64)
        }
    )
    label = features['label']
    print(label)
    img = features['features']
    print(img)
    # img = tf.decode_raw(img, tf.uint8)
    img = tf.reshape(img, [48, 48,1])
    # img = tf.reshape(img, [640, 480 ,3])
    # img = tf.cast(img, tf.float32)
          # * (1. / 255) - 0.5
    label = tf.cast(label, tf.int32)
    return img, label


#
img, label = loadData('./imgs/train')
def show_picture(X):
    plt.figure(figsize=(5, 5))
    plt.axis("off")
    plt.imshow(X,cmap='gray')
    plt.show()


img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                batch_size=1, capacity=20,
                                                min_after_dequeue=10)


sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

tf.train.start_queue_runners()

# from PIL import Image
# img=Image.open('/Users/apple/Documents/Python/DeepLearning/Driving_Detect/imgs2/train/1/img_115.jpg')
# im_array = np.array(img)
# print('====================================================')
# print(im_array)
# print('====================================================')
# print('====================================================')
# print(im_array.size)
# print(im_array.shape)
# show_picture(im_array)


# for i in range(10):
image,label = sess.run([img_batch,label_batch])
image = image.reshape((1,48,48))
print(image)
print(image.size)
print(image.shape)
print(image[0].shape)
print(type(image[0]))
print(label)

show_picture(image[0])
