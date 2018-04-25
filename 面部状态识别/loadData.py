'''
定义读取数据集的函数
'''

import tensorflow as tf

def loadData(path):
    # createData(path)
    # 创建文件队列,不限读取的数量
    filename_queue = tf.train.string_input_producer([path])
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
    img = features['features']
    img = tf.reshape(img, [48, 48, 1])
    label = tf.cast(label, tf.int32)
    return img, label