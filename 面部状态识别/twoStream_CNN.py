#!/usr/bin/env python3
# coding=utf-8

'''
双流卷积神经网络
包括两个卷积部分－－光流和RGB图像流
'''

_author_ = 'zixuwang'
_datetime_ = '2018-3-28'


import tensorflow as tf
tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES

import numpy as np


batch_size = 128
step = 100




s_image_holder = tf.placeholder(tf.float32,[batch_size,48,48,3])
t_image_holder = tf.placeholder(tf.float32,[batch_size,48,48,3])
label_holder = tf.placeholder(tf.int32,[batch_size])
keep_prob = tf.placeholder(tf.float32) #dropout时的保留率
learning_rate = tf.placeholder(tf.float32) #dropout时的保留率


# 权重的初始化...xavier初始化器
def weight_init(shape, name):
    weight = tf.get_variable(name+'wight',
                             shape=shape,
                             dtype=tf.float32,
                             initializer=tf.contrib.layers.xavier_initializer())
    return weight

# 偏置的初始化...用0.1防止神经元死亡
def bias_init(shape, name):
    bias = tf.get_variable(name+'bias',
                           shape=shape,
                           dtype=tf.float32,
                           initializer=tf.constant_initializer(0.1))

    return bias


# 卷积层初始化：［卷积核宽=3， 卷积核高=3， 输入通道数， 输出通道数］
def conv_layer(input,input_channels,output_channels,name):
    with tf.variable_scope(name) as scope:
        weight = weight_init([3,3,input_channels,output_channels],scope)
        bias = bias_init(output_channels,scope)
        kernel = tf.nn.conv2d(input=input,filter=weight,strides=[1,1,1,1],padding='SAME',name=scope+'kernel')
        conv = tf.nn.bias_add(kernel,bias,name=scope+'conv')
        conv_output = tf.nn.relu(conv,scope+'conv_output')
        return conv_output


# pooling层初始化：［池化核宽，池化核高］= [2, 2], 步长为 2
def pooling_layer(input,name):
    with tf.variable_scope(name) as scope:
        pool = tf.nn.max_pool(input,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name=scope+'pool')
        return pool

# 全链接层初始化
def fully_connect_layer(input,hidden_num,name):
    # 转换成一维向量
    dim_shape = input.get_shape()[-1].value
    with tf.variable_scope(name) as scope:
        weight = weight_init(shape=[dim_shape,hidden_num],name=scope)
        bias = bias_init(shape=[hidden_num],name=scope)
        # output = tf.nn.relu_layer(input,wight,bias,name=scope+'relu')
        output = tf.nn.relu(tf.nn.bias_add(tf.matmul(input, weight), bias), name=scope+'relu')
        return output

# logits层初始化
def logits_layer(input,hidden_num,name):
    # 转换成一维向量
    dim_shape = input.get_shape()[-1].value
    with tf.variable_scope(name) as scope:
        weight = weight_init(shape=[dim_shape,hidden_num],name=scope)
        bias = bias_init(shape=[hidden_num],name=scope)
        output = tf.nn.bias_add(tf.matmul(input, weight), bias, name=scope+'logits')
        return output


# 完整的卷积层和pooling层
def conv_and_pooling_layers(input_image,name):

    conv_1 = conv_layer(input=input_image,input_channels=3,output_channels=16,name=name+'/conv_1')
    pool_1 = pooling_layer(conv_1,name=name+'/pool_1')

    conv_2 = conv_layer(input=pool_1,input_channels=16,output_channels=32,name=name+'/conv_2')
    pool_2 = pooling_layer(conv_2,name=name+'/pool_2')

    conv_3 = conv_layer(input=pool_2,input_channels=32,output_channels=64,name=name+'/conv_3')
    pool_3 = pooling_layer(conv_3,name=name+'/pool_3')

    shape = pool_3.get_shape()
    dim = shape[1].value * shape[2].value * shape[3].value  # get dimension
    reshape = tf.reshape(pool_3, [-1, dim], name=name+'/reshape')

    local_1 = fully_connect_layer(input=reshape,hidden_num=256,name=name+'/local_1')
    local1_drop = tf.nn.dropout(local_1, keep_prob=keep_prob, name=name+'local_1_drop')
    # local_2 = fully_connect_layer(input=local1_drop,hidden_num=128,name=name+'/local_2')
    # local2_drop = tf.nn.dropout(local_2, keep_prob=keep_prob, name=name+'local_2_drop')


    return local1_drop


# 将空间卷积层和时间卷积层结合
def combine_layers(s_image,t_image):
    local_space = conv_and_pooling_layers(input_image=s_image,name='space')
    local_time = conv_and_pooling_layers(input_image=t_image,name='time')
    local_1 = local_space + local_time
    local_2 = fully_connect_layer(local_1,hidden_num=256,name='combine_1')
    local2_drop = tf.nn.dropout(local_2, keep_prob=keep_prob, name='combine_1_dropout')
    local_3 = fully_connect_layer(local2_drop,hidden_num=128,name='combine_2')
    local3_drop = tf.nn.dropout(local_3, keep_prob=keep_prob, name='combine_2_dropout')
    logits = logits_layer(local3_drop,hidden_num=6,name='logits')
    softmax = tf.nn.softmax(logits)
    prediction = tf.argmax(softmax, 1)

    return logits,softmax,prediction


# 定义损失函数
def loss(logits,labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits,name='cross_entropy_loss')
    cross_entropy_mean = tf.reduce_mean(cross_entropy_loss,name='cross_entropy_mean') #计算总的平均交叉熵损失
    tf.summary.scalar('Loss', cross_entropy_mean)
    return cross_entropy_mean


# 得到CNN输出、softmax输出以及预测结果
logits,softmax,prediction = combine_layers(s_image_holder,t_image_holder)

# 得到网络的损失
loss = loss(logits,label_holder)

# 定义优化器
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# 得到正确率
top_k_op = tf.nn.in_top_k(logits, label_holder,1,name='top_k_op')




# 读入数据
import loadData
train_img, train_label = loadData()
train_img_batch, train_label_batch = tf.train.shuffle_batch([train_img,train_label],
                                                batch_size=batch_size,capacity=2000,min_after_dequeue=128)


# 初始化
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# tensorboard合并
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("./logs/", sess.graph)

# 启动线程队列
tf.train.start_queue_runners()


# 开始训练
print('======> Start training2:')
for step in range(step):
    for i in range(2000):
        image_batch,label_batch = sess.run([train_img_batch,train_label_batch])    #得到一个batch的数据
        _,loss_value,logit,p,rs = sess.run([optimizer,loss,logits,prediction,merged],feed_dict={s_image_holder:image_batch,
                                                                                                t_image_holder:image_batch,
                                                                                                label_holder:label_batch,
                                                                                                keep_prob:0.8,
                                                                                                learning_rate:1e-3})
        print()
        print(p)
        print(label_batch)
        print(logit)

        if i % 10 == 0 and i != 0:
            print("PRINT!!!!!!")
            writer.add_summary(rs, i)

                                                                        #传入变量，开始训练
        format_str = ('step: %d, batch: %i  loss = %.2f')
        print(format_str % (step,i,loss_value))
    saver = tf.train.Saver()
    saver.save(sess=sess, save_path='./Saver/model_parameters.ckpt')
    print('Model saved')
print('======> Training End!!!!')

saver = tf.train.Saver()
saver.save(sess=sess, save_path='./Saver/model_parameters_final.ckpt')
print('Model saved')

