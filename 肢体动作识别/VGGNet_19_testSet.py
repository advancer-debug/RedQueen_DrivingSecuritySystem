#!/usr/bin/env python3
# coding=utf-8
'''

             ［VGGNet］
    构建VGGNet-19卷积神经网络
'''

_author_ = 'zixuwang'
_datetime_ = '2018-3-14'


import tensorflow as tf
tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES
import numpy as np
import vgg16
import loadData

# 图片尺寸：800*600 RGB
IMAGE_WIDTH = 640
IMAGE_HIGHT = 480
COLOR_CHANNELS = 3


# 设置每一个batch的大小
batch_size = 10
# 测试集总量
num_examples = 3000

# 加载VGG19模型以及设定其均值
VGG_Model = './model/VGG_19.mat'

# VGG 16 MODEL
VGG_16_Model = './model/VGG_16.mat'



'''
定义读取数据集的函数
'''



image_holder = tf.placeholder(tf.float32, [batch_size,IMAGE_HIGHT,IMAGE_WIDTH,COLOR_CHANNELS], name='image_holder')
label_holder = tf.placeholder(tf.int32,[batch_size], name='label_holder')
keep_prob = tf.placeholder(tf.float32, name='keep_prob') #dropout时的保留率


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

# 利用tf.nn.in_top_k计算输出结果top k的准确率，这里就是用top 1
top_k_op = tf.nn.in_top_k(logits, label_holder,1,name='top_k_op')

test_img,test_label = loadData(type='validation')
test_img_batch, test_label_batch = tf.train.batch([test_img,test_label],
                                                batch_size=batch_size, capacity=num_examples)


# 全局参数初始化
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# 读取模型参数
saver = tf.train.Saver()
model_path = "./VGGNet_19/model_parameters.ckpt"
saver.restore(sess, model_path)
print('CNN_model loaded')




# 这一步是启动图片数据增强的线程队列，一共使用16个线程来加速
# 如果不启动线程的话，后续的训练无法进行
tf.train.start_queue_runners()

#
# '''
# Step 6:Correct_prediction from testSet
# 对模型在测试集上进行准确率的验证
# 在对测试集验证的时候也需要采用批处理的方法验证
# '''
#
true_count = 0  #预测正确的次数
step = 0
print('test')

print('get data')

total_step = int(num_examples / batch_size)
while step<total_step:
    print(step)
    image_batch,label_batch = sess.run([test_img_batch,test_label_batch])   #取测试样本

    p = sess.run([top_k_op],feed_dict={image_holder:image_batch,label_holder:label_batch,keep_prob:1.0})#计算精度
     #这里的精度，也就是top_k_op，是指这批样本中预测类别和真实类别相等的情况
     #返回的是一个矩阵［1,1,0,0...1,0,1］，为0则代表预测错误，矩阵维度等于批处理样本的数量

    true_count += np.sum(p)    #计算这批样本中预测正确的个数
    print(true_count)
    step += 1

precision = true_count/num_examples*100   #计算精度（准确率）
print('测试集上面的精度为：%.2f%%'%precision)
