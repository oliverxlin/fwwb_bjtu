import tensorflow as tf
import numpy as np
import os, time, datetime, math
from tensorflow.contrib import learn
import network
import pandas as pd
from sklearn.utils import shuffle
import random
from keras.preprocessing.text import *
from keras.preprocessing import sequence
from keras.utils import to_categorical
import tensorflow as tf
import numpy as np
import os
import time
import datetime
from tensorflow.contrib import learn



FLAGS = tf.flags.FLAGS
# 特征数这里是第三级标签的特征数
max_features = 1192

# 句子填充的长度
sequence_len = 20

# batch 大小
batch_size = 512 

# 迭代次数
epochs = 10

is_training = True

# 词向量长度
embedding_dims = 24

# gru  的filters
num_filters = 4

# filter 的大小
filter_size = [1, 2, 3, 4, 5]

# 三个标签的类数
num_classes = [22, 191, 1192]

def predict(x_predict):
    # Training
    # ==================================================
        session_conf = tf.ConfigProto()
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        
        with sess.as_default():

            cnn = network.InceptionModel(sequence_length = sequence_len, 
                          num_classes = num_classes, 
                          embedding_size = embedding_dims, 
                          num_filters = num_filters)
            saver = cnn.saver
            saver = tf.train.import_meta_graph('model/inception_model-360.meta')
            saver.restore(sess, tf.train.latest_checkpoint("model/"))
            
            feed_dict = {
                  cnn.input_x: x_predict,
                  cnn.keep_prob: 1
            }
            predictions1, predictions2, predictions3 = sess.run(
                    [cnn.predictions1,cnn.predictions2, cnn.predictions3], feed_dict)
            
            print(predictions1)
if __name__ == '__main__':
    data = pd.read_csv('../data/train/processed_datay', sep=',')
    data = data[['label1', 'label2', 'label3']]
    data_x = np.load("../data/train/processed_datax.npy")
    data["ids"] = data_x

    # train_data = data.sample(frac= 0.1).reset_index()
    # test = data.sample(frac= 0.1).reset_index()
    train_data = data[0:20]
    test = data[0:20]
    train_y = train_data[['label1', 'label2', 'label3']]
    test_y = test[['label1', 'label2', 'label3']]

    train_x = train_data["ids"]
    test_x = test["ids"]

    train_x = sequence.pad_sequences(train_x, maxlen=20)
    test_x  = sequence.pad_sequences(test_x, maxlen=20) 



    # 先生成唯一数组
    y_label1 = data['label1'].unique().tolist()
    y_label2 = data['label2'].unique().tolist()
    y_label3 = data['label3'].unique().tolist()

    # 获取在唯一数组中的索引(训练集和测试集各有3个标签需要处理)
    train_y_label1_map = train_y['label1'].apply(lambda x: y_label1.index(x))
    train_y_label2_map = train_y['label2'].apply(lambda x: y_label2.index(x))
    train_y_label3_map = train_y['label3'].apply(lambda x: y_label3.index(x))
    test_y_label1_map = test_y['label1'].apply(lambda x: y_label1.index(x))
    test_y_label2_map = test_y['label2'].apply(lambda x: y_label2.index(x))
    test_y_label3_map = test_y['label3'].apply(lambda x: y_label3.index(x))

        # 生成对应one-hot(用做训练模型的标签)
    train_y_label1_one_hot = to_categorical(train_y_label1_map, 22)
    train_y_label2_one_hot = to_categorical(train_y_label2_map, 191)
    train_y_label3_one_hot = to_categorical(train_y_label3_map, 1192)
    test_y_label1_one_hot = to_categorical(test_y_label1_map, 22)
    test_y_label2_one_hot = to_categorical(test_y_label2_map, 191)
    test_y_label3_one_hot = to_categorical(test_y_label3_map, 1192)

    y_train = [train_y_label1_one_hot,train_y_label2_one_hot,train_y_label3_one_hot]
    y_test = [test_y_label1_one_hot,test_y_label2_one_hot,test_y_label3_one_hot]
    predict(train_x)
    print(np.argmax(train_y_label1_one_hot, axis = 1))