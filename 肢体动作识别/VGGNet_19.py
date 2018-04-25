#!/usr/bin/env python3
# coding=utf-8
'''

             ［VGGNet］
    构建VGGNet-19卷积神经网络

Tensorboard: http://wiki.jikexueyuan.com/project/tensorflow-zh/api_docs/python/train.html#MomentumOptimizer
'''

_author_ = 'zixuwang & jiachenxu'
_datetime_ = '2018-3-25'


import tensorflow as tf
tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES
import numpy as np
import vgg16
import loadData


# config = tf.ConfigProto(allow_soft_placement=True)
# #最多占gpu资源的70%
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
# #开始不会给tensorflow全部gpu资源 而是按需增加
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
TRAINING_DATA = './imgs/train'
VALIDATION_IMAGE = './imgs/validation'
TEST_IMAGE = './images/test'

# 图片尺寸：640*480 RGB
IMAGE_WIDTH = 640
IMAGE_HIGHT = 480
COLOR_CHANNELS = 3

# 设置总的训练轮数
total_step = 30
# 设置每一个batch的大小
batch_size = 12
capacity = 2000
min_after_dequeue = 16
# 测试集总量
num_examples = 3000

# 加载VGG19模型以及设定其均值
# VGG_19_Model = './model/VGG_19.mat'

# VGG 16 MODEL
VGG_16_Model = './model/VGG_16.mat'


image_holder = tf.placeholder(tf.float32, [batch_size,IMAGE_HIGHT,IMAGE_WIDTH,COLOR_CHANNELS], name='image_holder')
label_holder = tf.placeholder(tf.int32,[batch_size], name='label_holder')
keep_prob = tf.placeholder(tf.float32, name='keep_prob') #dropout时的保留率
learning_rate = tf.placeholder(tf.float32, name='learning_rate')



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
    weights = tf.get_variable(name+'w',shape=[num_input,num_hidden_layer],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
    # tf.summary.histogram(name+'w',weights)
    #初始化0.2防止dead neuron
    bias = tf.Variable(tf.constant(0.2,shape=[num_hidden_layer],dtype=tf.float32,name=name+'b'))
    # tf.summary.histogram(name+'b', bias)

    # output = tf.nn.relu(tf.nn.bias_add(tf.matmul(input,weights),bias),name=name)
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
def VGGNet(path,keep_prob):
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

'''
Step 3:Loss
计算损失，构建损失函数
本次使用cross-entropy Loss

==> loss = - Sum(y_ * log(y))


注意
reduce_sum、reduce_mean分别是求和、求平均
reduction_indices ＝ 1（或［1，0］）的意思是，将得到的结果按照行压缩求和、求平均
reduction_indices ＝ 0（或［0，1］）的意思是，将得到的结果按照列压缩求和、求平均
如果没有这个参数就意味着把矩阵弄成一个数字
具体见下图：
https://pic3.zhimg.com/v2-c92ac5c3a50e4bd3d60e29c2ddc4c5e9_r.jpg
'''
def loss(logits, labels):
    labels = tf.cast(labels,tf.int64)
    # label = tf.one_hot(indices=labels,depth=10)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits( logits=logits,
                                                                    labels=labels,
                                                                    name='cross_entropy_per_example')
    # 计算每一个样本的交叉熵损失 （对输出进行softmax）
    cross_entropy_mean = tf.reduce_mean(cross_entropy,name='cross_entropy') #计算总的平均交叉熵损失
    # cross_entropy_mean =  tf.reduce_mean(-tf.reduce_sum(label * tf.log(logits), reduction_indices=[1]),name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy_mean)
    return cross_entropy_mean  #返回所有样本的总损失


'''
Step 4:Optimizer
选择优化器，并指定优化器优化loss
主要是用梯度下降法、随机梯度下降法、批处理梯度下降法
选择Adam优化器，设置学习率（a=1e-3）
TF会自动进行BP算法梯度更新
'''


prediction,softmax,logits = VGGNet(path=VGG_16_Model,keep_prob=keep_prob)
loss = loss(logits,label_holder)    #将神经网络输出结果和真实标记传入loss函数得到总损失
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
# train_step = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(loss)
# train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
# train_step = tf.train.AdagradOptimizer(learning_rate).minimize(loss)
# 利用tf.nn.in_top_k计算输出结果top k的准确率，这里就是用top 1
top_k_op = tf.nn.in_top_k(logits, label_holder,1,name='top_k_op')



train_img, train_label = loadData()
train_img_batch, train_label_batch = tf.train.shuffle_batch([train_img,train_label],
                                                batch_size=batch_size,capacity=capacity,min_after_dequeue=min_after_dequeue)




test_img,test_label = loadData(type='validation')
test_img_batch, test_label_batch = tf.train.batch([test_img,test_label],
                                                batch_size=1, capacity=num_examples)
'''
Step 4:Training
开始训练，使用批处理梯度下降、每次选一个mini_batch，并feed给placeholder
(总共30轮，每个batch包含128样本)
当然在一开始的时候需要调用TF全局参数初始化器
InteractiveSession是将这个session设置为默认session
'''
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()


merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("./logs/", sess.graph)


# 这一步是启动图片数据增强的线程队列，一共使用16个线程来加速
# 如果不启动线程的话，后续的训练无法进行
tf.train.start_queue_runners()
print('======> Start training2:')
for step in range(total_step):
    for i in range(2000):
        lr = 1e-4
        image_batch,label_batch = sess.run([train_img_batch,train_label_batch])    #得到一个batch的数据
        _,loss_value,logit,p,rs = sess.run([train_step,loss,logits,prediction,merged],feed_dict={image_holder:image_batch,label_holder:label_batch,keep_prob:0.8,learning_rate:lr})
        print()
        print(lr)
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
    saver.save(sess=sess, save_path='./VGGNet_19/model_parameters.ckpt')
    print('Model saved')
print('======> Training End!!!!')

saver = tf.train.Saver()
saver.save(sess=sess, save_path='./VGGNet_19/model_parameters_final.ckpt')
print('Model saved')

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


while step<num_examples:
    print(step)
    image_batch,label_batch = sess.run([test_img_batch,test_label_batch])   #一次取一个测试样本
    print('get_img')
    prediction = sess.run([top_k_op],feed_dict={image_holder:image_batch,label_holder:label_batch,keep_prob:1.0})#计算精度
     #这里的精度，也就是top_k_op，是指这批样本中预测类别和真实类别相等的情况
     #返回的是一个矩阵［1,1,0,0...1,0,1］，为0则代表预测错误，矩阵维度等于批处理样本的数量

    true_count += np.sum(prediction)    #计算这批样本中预测正确的个数

    step += 1

precision = true_count/num_examples*100   #计算精度（准确率）
print('测试集上面的精度为：%.2f%%'%precision)
