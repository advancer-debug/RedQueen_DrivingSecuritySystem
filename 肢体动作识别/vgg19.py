import tensorflow as tf
import scipy.io as sio
import numpy as np
import scipy.misc

_author_ = 'zixuwang'
_datetime_ = '2018-3-25'
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
def build_vgg19(path, image_holder):
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


