#!/usr/bin/env python3
# coding=utf-8

'''
卷积神经网络

本神经网络为bagging集成的面部微表情识别第一个网络
4个conv－pooling层
3个全链接层
1个softmax层

除此之外，考虑到每批样本数量很多，使用了BN防止梯度弥散

BN通俗解释：https://morvanzhou.github.io/tutorials/machine-learning/ML-intro/3-08-batch-normalization/#每层都做标准化

BN代码解释：https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-13-BN/

'''

_author_ = 'zixuwang'
_datetime_ = '2018-3-28'


import tensorflow as tf
tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES

import numpy as np


batch_size = 1
step = 3589
on_train = tf.cast(False,dtype=tf.bool)



image_holder = tf.placeholder(tf.float32,[batch_size,48,48,1])
label_holder = tf.placeholder(tf.int32,[batch_size])
# keep_prob = tf.placeholder(tf.float32) #dropout时的保留率
learning_rate = tf.placeholder(tf.float32) #dropout时的保留率


# 权重的初始化...xavier初始化器
def weight_init(shape, scope):
    weight = tf.get_variable(scope+'wight',
                             shape=shape,
                             dtype=tf.float32,
                             initializer=tf.contrib.layers.xavier_initializer())
    return weight

# 偏置的初始化...用0.1防止神经元死亡
def bias_init(shape, scope):
    bias = tf.get_variable(scope+'bias',
                           shape=shape,
                           dtype=tf.float32,
                           initializer=tf.constant_initializer(0.1))

    return bias

# BN
# def BatchNormalization(input,output_size,scope,type):
#     return input
def BatchNormalization(input,output_size,scope,type):
    if type == 'conv':
        shape = [0,1,2]
    else:
        shape = [0]
    fc_mean, fc_var = tf.nn.moments(input,
                                    axes=shape  # 想要 normalize 的维度, [0] 代表 batch 维度
                                    )  # 如果是图像数据, 可以传入 [0, 1, 2], 相当于求[batch, height, width] 的均值/方差, 注意不要加入 channel 维度
    scale = tf.Variable(tf.ones([output_size]))
    shift = tf.Variable(tf.zeros([output_size]))
    epsilon = 0.001

    ema = tf.train.ExponentialMovingAverage(decay=0.5)  # exponential moving average 的 decay 度

    # 由于每个batch都会有BN，所以均值/方差应该是一个全局存在的,所以要把这些信息记录下来
    def mean_var_with_update():
        ema_apply_op = ema.apply([fc_mean, fc_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(fc_mean), tf.identity(fc_var)

    mean, var = tf.cond(on_train,  # on_train 的值是 True/False
                        mean_var_with_update,  # 如果是 True, 更新 mean/var
                        lambda: (  # 如果是 False, 返回之前 fc_mean/fc_var 的Moving Average
                            ema.average(fc_mean),
                            ema.average(fc_var)
                        )
                        )  # 根据新的 batch 数据, 记录并稍微修改之前的 mean/var

    bn_input = tf.nn.batch_normalization(input, mean=mean, variance=var, offset=shift, scale=scale,
                                     variance_epsilon=epsilon, name=scope + 'BN')
    # Wx_plus_b = (Wx_plus_b - fc_mean) / tf.sqrt(fc_var + 0.001)
    # Wx_plus_b = Wx_plus_b * scale + shift
    return bn_input




# 卷积层初始化：［卷积核宽=3， 卷积核高=3， 输入通道数， 输出通道数］
def conv_layer(input,input_channels,output_channels,name,ifBN = True):
    with tf.name_scope(name) as scope:
        weight = weight_init([3,3,input_channels,output_channels],scope=scope)
        bias = bias_init(output_channels,scope=scope)
        kernel = tf.nn.conv2d(input=input,filter=weight,strides=[1,1,1,1],padding='SAME',name=scope+'kernel')
        conv = tf.nn.bias_add(kernel,bias,name=scope+'conv')
        if ifBN == True:
            conv = BatchNormalization(conv,output_channels,scope,'conv')
        conv_output = tf.nn.relu(conv,scope+'conv_output')
        return conv_output


# pooling层初始化：［池化核宽，池化核高］= [2, 2], 步长为 2
def pooling_layer(input,name):
    with tf.name_scope(name) as scope:
        pool = tf.nn.max_pool(input,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name=scope+'pool')
        return pool

# 全链接层初始化
def fully_connect_layer(input,hidden_num,name,ifBN = True):
    # 转换成一维向量
    dim_shape = input.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        weight = weight_init(shape=[dim_shape,hidden_num],scope=scope)
        bias = bias_init(shape=[hidden_num],scope=scope)
        # output = tf.nn.relu_layer(input,wight,bias,name=scope+'relu')
        kernel = tf.nn.bias_add(tf.matmul(input, weight), bias,name=scope+'kernel')
        if ifBN == True:
            kernel = BatchNormalization(kernel,hidden_num,scope,'full')
        output = tf.nn.relu(kernel, name=scope+'relu')
        return output

# logits层初始化(不做BN)
def logits_layer(input,hidden_num,name):
    # 转换成一维向量
    dim_shape = input.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        weight = weight_init(shape=[dim_shape,hidden_num],scope=scope)
        bias = bias_init(shape=[hidden_num],scope=scope)
        output = tf.nn.bias_add(tf.matmul(input, weight), bias, name=scope+'logits')
        return output


# 完整的卷积层和pooling层
def conv_and_pooling_layers(input_image):

    input_image = BatchNormalization(input_image,1,scope='',type='conv')
    conv_1 = conv_layer(input=input_image,input_channels=1,output_channels=16,name='conv_1')
    pool_1 = pooling_layer(conv_1,name='pool_1')

    conv_2 = conv_layer(input=pool_1,input_channels=16,output_channels=32,name='conv_2')
    pool_2 = pooling_layer(conv_2,name='pool_2')

    conv_3 = conv_layer(input=pool_2,input_channels=32,output_channels=64,name='conv_3')
    pool_3 = pooling_layer(conv_3,name='pool_3')

    # conv_4 = conv_layer(input=pool_3,input_channels=64,output_channels=128,name='conv_4')
    # pool_4 = pooling_layer(conv_4,name='pool_4')

    shape = pool_3.get_shape()
    dim = shape[1].value * shape[2].value * shape[3].value  # get dimension
    reshape = tf.reshape(pool_3, [-1, dim], name='reshape')

    local_1 = fully_connect_layer(input=reshape,hidden_num=512,name='local_1')
    # local1_drop = tf.nn.dropout(local_1, keep_prob=keep_prob, name='local_1_drop')
    local_2 = fully_connect_layer(input=local_1,hidden_num=512,name='local_2')
    # local2_drop = tf.nn.dropout(local_2, keep_prob=keep_prob, name='local_2_drop')
    local_3 = fully_connect_layer(local_2, hidden_num=512, name='local_3')
    # local3_drop = tf.nn.dropout(local_3, keep_prob=keep_prob, name='local_3_drop')
    logits = logits_layer(local_3, hidden_num=7, name='logits')
    softmax = tf.nn.softmax(logits)
    prediction = tf.argmax(softmax, 1)
    return logits,softmax,prediction

# 得到CNN输出、softmax输出以及预测结果
logits,softmax,prediction = conv_and_pooling_layers(image_holder)


# 读入数据
from loadData import loadData
test_img, test_label = loadData('./DataSet/TestSet.tfrecords')
test_img_batch, test_label_batch = tf.train.shuffle_batch([test_img,test_label],
                                                batch_size=batch_size,capacity=2000,min_after_dequeue=1)

# 初始化
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# 读取模型参数
saver = tf.train.Saver()
model_path = "./Saver/model_parameters.ckpt"
saver.restore(sess, model_path)
print('CNN_model loaded')


# 启动线程队列
tf.train.start_queue_runners()


photoDict = {
    0:'生 气',
    1:'沮 丧',
    2:'害 怕',
    3:'开 心',
    4:'悲 伤',
    5:'吃 惊',
    6:'正 常',
}



import matplotlib.pyplot as plt
# 开始预测
print('======> Start :')
for i in range(step):
    img,label = sess.run([test_img_batch,test_label_batch])
    p = sess.run([prediction],feed_dict={image_holder:img,label_holder:label})
    print('真实类别： ', photoDict[label[0]])
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(4,4))
    plt.axis('off')
    plt.imshow(img[0].reshape((48,48)),cmap='gray')
    plt.title('预测类别： '+ photoDict[p[0][0]])

    plt.show()


