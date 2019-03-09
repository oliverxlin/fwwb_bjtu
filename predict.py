import tensorflow as tf
import sys
sys.path.append("networks/")
import inception_model.network as network
import textcnn_model.network as tt
import os, time, datetime, math
from tensorflow.contrib import learn
import pandas as pd
from sklearn.utils import shuffle
import random
from keras.preprocessing.text import *
from keras.preprocessing import sequence
from keras.utils import to_categorical
import numpy as np
import os
from tensorflow.contrib import learn
class Predict:
    def __init__(self, metapath, cppath):

        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.graph= tf.Graph()#为每个类(实例)单独创建一个graph
        self.sess = tf.Session(graph = self.graph, config= config)
        with self.graph.as_default():
            # self.saver = tf.train.Saver()
            self.saver=tf.train.import_meta_graph(metapath)#创建恢复器
            self.saver.restore(self.sess,tf.train.latest_checkpoint(cppath))
            self.X = self.graph.get_tensor_by_name("Input/input_x:0")
            self.Prob = self.graph.get_tensor_by_name("Input/keep_prob:0")
            self.pre1 = self.graph.get_tensor_by_name("output1/predictions:0")
            self.pre2 = self.graph.get_tensor_by_name("output2/predictions:0")
            self.pre3 = self.graph.get_tensor_by_name("output3/predictions:0")
        # self.X = model.input_x
            # self.Prob = model.keep_prob
        # self.model = model

    def predict(self, data):
        feed_dict = {
                self.X :data,
                self.Prob: 1
        }
        with self.sess.as_default():
            predictions1, predictions2, predictions3 = self.sess.run(
                        [self.pre1,
                        self.pre2, 
                        self.pre3], feed_dict)

        return predictions1, predictions2, predictions3

if __name__ == '__main__':

    predict = Predict("networks/textcnn_model/model/textcnn_model-2120.meta",
                      "networks/textcnn_model/model/",
                      )
    predict2 = Predict("networks/inception_model/model/inception_model-11960.meta",
                      "networks/inception_model/model/",
                      )      

    data = pd.read_csv('data/train/processed_datay', sep=',')
    data = data[['label1', 'label2', 'label3']]
    data_x = np.load("data/train/processed_datax.npy")
    data["ids"] = data_x

    # train_data = data.sample(frac= 0.1).reset_index()
    # test = data.sample(frac= 0.1).reset_index()
    train_data = data[10:30]
    test = data[10:30]
    train_y = train_data[['label1', 'label2', 'label3']]
    test_y = test[['label1', 'label2', 'label3']]

    train_x = train_data["ids"]
    test_x = test["ids"]

    train_x = sequence.pad_sequences(train_x, maxlen=20)
    test_x  = sequence.pad_sequences(test_x, maxlen=20) 



    # 先生成唯一数组
    y_label1 = []
    y_label2 = []
    y_label3 = []
    with open("data/label1.txt","r") as f:
        y_label1 = f.read().split(' ')
    with open("data/label2.txt","r") as f:
        y_label2 = f.read().split(' ')
    with open("data/label3.txt","r") as f:
        y_label3 = f.read().split(' ')



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

    print(predict.predict(train_x)[0])
    print(predict2.predict(train_x)[0])
    print(np.argmax(train_y_label1_one_hot, axis = 1))