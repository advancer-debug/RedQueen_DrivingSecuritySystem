import tensorflow as tf

def loadData(type = 'train'):
    # createData(path)
    # 创建文件队列,不限读取的数量
    if type == 'train':
        fileName = '00001Train.tfrecords'
    elif type == 'test':
        fileName = '00001Test.tfrecords'
    else:
        fileName = '00001Validation.tfrecords'
    print(fileName)
    filename_queue = tf.train.string_input_producer([fileName])
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
            'img_raw': tf.FixedLenFeature([], tf.string)
        }
    )
    label = features['label']
    img = features['img_raw']
    img = tf.decode_raw(img, tf.uint8)
    img = tf.reshape(img, [IMAGE_HIGHT, IMAGE_WIDTH, COLOR_CHANNELS])

    # img = tf.cast(img, tf.float32) - MEAN_VALUE
          # * (1. / 255)
    # label = tf.cast(label, tf.int32)
    return img, label