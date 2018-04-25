#!/usr/bin/env python3
# coding=utf-8
'''

             ［VGGNet］
    构建VGGNet-19卷积神经网络
'''

_author_ = 'zixuwang & jiachenxu'
_datetime_ = '2018-3-25'


import tensorflow as tf
tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES
import time
import numpy as np
import scipy.io
import scipy.misc
import vgg16
from PIL import Image
import os
import matplotlib.pyplot as plt
import random


# 图片尺寸：800*600 RGB
IMAGE_WIDTH = 640
IMAGE_HIGHT = 480
COLOR_CHANNELS = 3


# 设置每一个batch的大小
batch_size = 1
# 测试集总量
num_examples = 3000

# 加载VGG19模型以及设定其均值
VGG_Model = './model/VGG_19.mat'

# VGG 16 MODEL
VGG_16_Model = './model/VGG_16.mat'



'''
定义读取数据集的函数
'''
def loadData(img_path):
    img = scipy.misc.imread(img_path)
    img = np.reshape(img,(1,)+img.shape)
    # img = Image.open(img_path)
    # img = img.resize((IMAGE_WIDTH, IMAGE_HIGHT))
    return img

image_holder = tf.placeholder(tf.float32, [batch_size,IMAGE_HIGHT,IMAGE_WIDTH,COLOR_CHANNELS], name='image_holder')
keep_prob = tf.placeholder(tf.float32, name='keep_prob') #dropout时的保留率

'''
＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊
                                        重构VGGNet：迁移学习
＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊
'''

# 建立卷积层和池化层
def build_net(type, input, weights_and_biases=None):
    if type == 'conv':
        return tf.nn.relu(tf.nn.conv2d(input,weights_and_biases[0],strides=[1,1,1,1],padding='SAME') + weights_and_biases[1])

    elif type == 'pool':
        return tf.nn.max_pool(input,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


# 返回网络某一卷积层的权重和偏置
def get_weight_bias(vgg_layers, i):
    weights = vgg_layers[i][0][0][0][0][0]
    weights = tf.constant(weights)

    biases = vgg_layers[i][0][0][0][0][1]
    biases = tf.constant(np.reshape(biases,(biases.size)))

    return weights,biases

# 提取.mat中模型，重构网络卷积部分
def build_vgg19(path):
    net = {}
    # 从磁盘读取模型
    vgg_rawnet = scipy.io.loadmat(path)
    vgg_layers = vgg_rawnet['layers'][0]
    # net['input'] = tf.Variable(np.zeros((IMAGE_NUM,IMAGE_HIGHT,IMAGE_WIDTH,COLOR_CHANNELS)).astype('float32'))

    net['conv1_1'] = build_net('conv',image_holder, get_weight_bias(vgg_layers, 0))
    net['conv1_2'] = build_net('conv', net['conv1_1'], get_weight_bias(vgg_layers, 2))
    net['pool1'] = build_net('pool', net['conv1_2'])

    net['conv2_1'] = build_net('conv', net['pool1'], get_weight_bias(vgg_layers, 5))
    net['conv2_2'] = build_net('conv', net['conv2_1'], get_weight_bias(vgg_layers, 7))
    net['pool2'] = build_net('pool', net['conv2_2'])

    net['conv3_1'] = build_net('conv', net['pool2'], get_weight_bias(vgg_layers, 10))
    net['conv3_2'] = build_net('conv', net['conv3_1'], get_weight_bias(vgg_layers, 12))
    net['conv3_3'] = build_net('conv', net['conv3_2'], get_weight_bias(vgg_layers, 14))
    net['conv3_4'] = build_net('conv', net['conv3_3'], get_weight_bias(vgg_layers, 16))
    net['pool3'] = build_net('pool', net['conv3_4'])

    net['conv4_1'] = build_net('conv', net['pool3'], get_weight_bias(vgg_layers, 19))
    net['conv4_2'] = build_net('conv', net['conv4_1'], get_weight_bias(vgg_layers, 21))
    net['conv4_3'] = build_net('conv', net['conv4_2'], get_weight_bias(vgg_layers, 23))
    net['conv4_4'] = build_net('conv', net['conv4_3'], get_weight_bias(vgg_layers, 25))
    net['pool4'] = build_net('pool', net['conv4_4'])

    net['conv5_1'] = build_net('conv', net['pool4'], get_weight_bias(vgg_layers, 28))
    net['conv5_2'] = build_net('conv', net['conv5_1'], get_weight_bias(vgg_layers, 30))
    net['conv5_3'] = build_net('conv', net['conv5_2'], get_weight_bias(vgg_layers, 32))
    net['conv5_4'] = build_net('conv', net['conv5_3'], get_weight_bias(vgg_layers, 34))
    net['pool5'] = build_net('pool', net['conv5_4'])

    return net



'''
***********************************************************************************************************
***********************************************************************************************************
                                            VGGNet全连接层
***********************************************************************************************************
***********************************************************************************************************

'''

# 全连接层［］
def fullyconnect_layer(input,num_hidden_layer,name,type='full'):
    num_input = input.get_shape()[-1].value
    # weights = tf.Variable(tf.random_normal([num_input,num_hidden_layer],mean=0.0),dtype=tf.float32,name=name+'w')
    weights = tf.get_variable(name+'w',shape=[num_input,num_hidden_layer],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
    # tf.summary.histogram(name+'w',weights)
    # bias = tf.get_variable(name+'b',shape=[num_hidden_layer],dtype=tf.float32,initializer=0.1)       #初始化0.1防止dead neuron
    bias = tf.Variable(tf.constant(0.2,shape=[num_hidden_layer],dtype=tf.float32,name=name+'b'))
    # tf.summary.histogram(name+'b', bias)

    # output = tf.nn.relu(tf.nn.bias_add(tf.matmul(input,weights),bias),name=name)
    print(type)
    if type =='full':
        # output = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(input, weights), bias), name=name)
        output = tf.nn.relu(tf.nn.bias_add(tf.matmul(input,weights),bias),name=name)
        # output = tf.nn.relu_layer(input,weights,bias,name=name)
    else:
        output = tf.nn.bias_add(tf.matmul(input,weights),bias,name=name)

    return output



'''
Step 2:FeedForward
定义算法公式，就是神经网络前向传播的途径
TF会自动帮忙计算反向传播的梯度公式
首先先进行一些函数的定义，方便后面使用
'''
def VGGNet(path):
    # 卷积、池化层
    # vgg19 = build_vgg19(path)

    vgg16_net = vgg16.vgg16_conv_net(file_path=path, input_image=image_holder)

    # 全连接层
    # Layer6:       local1:  ［?，256］
    #               local2:  ［256,256］
    conv_output = vgg16_net['pool5']
    # conv_output = vgg19['pool5']
    print(conv_output)
    shape = conv_output.get_shape()
    print(shape)
    dim = shape[1].value * shape[2].value * shape[3].value      #dimension reduce
    reshape = tf.reshape(conv_output,[-1,dim],name='reshape')
    print(reshape)

    local1 = fullyconnect_layer(reshape,num_hidden_layer=712,name='local1')
    local1_drop = tf.nn.dropout(local1,keep_prob=keep_prob,name='local1_drop')
    local2 = fullyconnect_layer(local1_drop,num_hidden_layer=468,name='local2')

    local2_drop = tf.nn.dropout(local2,keep_prob=keep_prob,name='local2_drop')


    # 最后一层输出层,10类
    # 暂时不需要连接一个softmax，到计算损失的时候再用softmax
    # 因为不用softmax也可以比较出最大的那一个作为输出，得到分类结果
    # softmax只是用来计算loss
    # 不采用激活函数，不采用dropout
    logits = fullyconnect_layer(local2_drop,num_hidden_layer=10,name='logits',type='out')

    # w = tf.Variable(tf.truncated_normal([256,10],stddev=1/256.0))
    # b = tf.Variable(tf.constant(0.0,shape=[10]))
    # logits = tf.add(tf.matmul(local2_drop, w),b,name='logits')


    softmax = tf.nn.softmax(logits,name='softmax')
    prediction = tf.argmax(softmax, 1, name='prediction')
    return prediction,softmax,logits

# 本次测试用不到优化器、损失函数等

'''
***********************************************************************************************************
***********************************************************************************************************
                                                VGGNet定义完毕
***********************************************************************************************************
***********************************************************************************************************

'''



prediction,softmax,logits = VGGNet(path=VGG_16_Model)



'''
Step 4:Training
开始训练，使用批处理梯度下降、每次选一个mini_batch，并feed给placeholder
(总共30轮，每个batch包含128样本)
当然在一开始的时候需要调用TF全局参数初始化器
InteractiveSession是将这个session设置为默认session
'''
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# 读取模型参数
saver = tf.train.Saver()
model_path = "./VGGNet_19/model_parameters.ckpt"
saver.restore(sess, model_path)
print('CNN_model loaded')


# step = 0
print('test')

print('get data')


photoDict = {
    0:'正 常 驾 驶',
    1:'右 手 玩 手 机',
    2:'右 手 打 电 话',
    3:'左 手 玩 手 机',
    4:'左 手 打 电 话',
    5:'调 节 收 音 机',
    6:'喝 饮 料 吃 东 西',
    7:'肢 体 非 正 常 扭 转',
    8:'整 理 头 发 化 妆',
    9:'转 头 说 话',

}
path = './test'






import cv2
import opencv

cap = cv2.VideoCapture(0)
while True:
    img = opencv.getPhoto(cap)

    img = np.reshape(img,(1,)+img.shape)
    cv2.imwrite('1.png', img[0])
    p = sess.run([prediction],feed_dict={image_holder:img,keep_prob:1.0})#计算精度
    # print(type(p[0][0]))
    # print(photoDict[int(file.split('#')[1].split('.')[0])])
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(8,6))
    plt.axis('off')
    tmp = plt.imread('1.png')
    plt.imshow(tmp)
    plt.title(photoDict[p[0][0]])
    plt.show()
    cv2.waitKey()

# step += 1


cap.release()
cv2.destroyAllWindows()

tmp = plt.imread('1.png')
plt.imshow(tmp)
plt.show()