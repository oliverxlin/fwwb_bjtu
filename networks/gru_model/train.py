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
from tqdm import tqdm,trange
FLAGS = tf.flags.FLAGS
# 特征数这里是第三级标签的特征数
max_features = 1192

# 句子填充的长度
sequence_len = 20

# batch 大小
batch_size = 256 

# 迭代次数
epochs = 5

is_training = True

# 词向量长度
embedding_dims = 24

# gru  的filters
num_filters = 32

# filter 的大小
filter_size = [1, 2, 3, 4, 5]

# 三个标签的类数
num_classes = [22, 191, 1192]

# batch 生成函数
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
  
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = np.min(((batch_num + 1) * batch_size, data_size))
            yield shuffled_data[start_index:end_index]

def train(x_train, y_train, x_dev, y_dev):

    session_conf = tf.ConfigProto()
    
    session_conf.gpu_options.allow_growth = True
    sess = tf.Session(config=session_conf)

    with sess.as_default():
        cnn = network.Gru(sequence_length = sequence_len, 
                          num_classes = num_classes, 
                          embedding_size = embedding_dims, 
                          num_filters = num_filters)
        saver= cnn.saver
        # Define Training procedure

        global_step = tf.Variable(0, name="global_step", trainable=False)

        #学习率衰减
        learning_rate = tf.train.exponential_decay(
            0.01,
            global_step,
            x_train.shape[0] / batch_size,
            0.99,
            staircase=True,)

        # 优化器
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # 参数初始化
        sess.run(tf.global_variables_initializer())

        merged = tf.summary.merge_all()

        writer = tf.summary.FileWriter("fwwb",sess.graph)
        acc_sum = np.array([0.0,0.0,0.0,0.0])
        loss_sum = 0

        def train_step(x_batch, y_batch1, y_batch2, y_batch3, loss_sum, acc_sum, interal = 10):

            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y1: y_batch1,
              cnn.input_y2: y_batch2,
              cnn.input_y3: y_batch3,
              cnn.keep_prob: 0.8
            }

#         三个标签单独的准确率， 一起的准确率共四个准确率
            rs, _, step,  loss, accuracy1, accuracy2, accuracy3, acc = sess.run(
                [merged, train_op, global_step, cnn.loss, cnn.accuracy1, cnn.accuracy2, cnn.accuracy3, cnn.acc],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            acc_sum += [accuracy1, accuracy2, accuracy3, acc]
            loss_sum += loss
            if step % interal == 0:
                saver.save(sess, "model/gru_model", global_step=step)
                print("{}: step {}, loss {:g}, acc {}".format(time_str, step, loss_sum/interal, acc_sum/interal))
                loss_sum = 0
                acc_sum = np.array([0.0,0.0,0.0,0.0])
                writer.add_summary(rs, step)
            return loss_sum, acc_sum
        # 评估步骤
        def dev_step(x_batch, y_batch):
                """
                Evaluates model on a dev set
                """

                dev_batches = batch_iter(
                    list(zip(x_batch, y_batch[0], y_batch[1], y_batch[2])), batch_size, 1)
                step = 0
                loss = 0
                accuracy = np.array([0.0, 0.0, 0.0,0.0])
                for batch in dev_batches:
                    x_batch_dev, y_batch1_dev, y_batch2_dev, y_batch3_dev = zip(*batch)
                    feed_dict = {
                      cnn.input_x: x_batch_dev,
                      cnn.input_y1: y_batch1_dev,
                      cnn.input_y2: y_batch2_dev,
                      cnn.input_y3: y_batch3_dev,
                      cnn.keep_prob:1
                    }
                    if len(x_batch_dev) < batch_size:
                        continue
                    step,  temp_loss, temp_accuracy1, temp_accuracy2, temp_accuracy3 , acc= sess.run(
                        [global_step,cnn.loss, cnn.accuracy1, cnn.accuracy2, cnn.accuracy3, cnn.acc],
                        feed_dict)
                    accuracy[0] += temp_accuracy1 * len(x_batch_dev)
                    accuracy[1] += temp_accuracy2 * len(x_batch_dev)
                    accuracy[2] += temp_accuracy3 * len(x_batch_dev)
                    accuracy[3] += acc * len(x_batch_dev)
                    loss += temp_loss * len(x_batch_dev)
                accuracy /= x_batch.shape[0]
                loss /= x_batch.shape[0]
                time_str = datetime.datetime.now().isoformat()                    
                print("Evaluation:")
                print("{}: step {}, loss {:g}, acc {} ".format(time_str, step, loss, accuracy))


        # Generate batches
        batches = batch_iter(
            list(zip(x_train, y_train[0],y_train[1], y_train[2] )), batch_size, epochs)
        # Training loop. For each batch...
        for batch in tqdm(batches):
            x_batch, y_batch1, y_batch2, y_batch3 = zip(*batch)
            if len(x_batch) < batch_size:
                continue
            loss_sum, acc_sum = train_step(x_batch, y_batch1, y_batch2, y_batch3,  loss_sum, acc_sum, 40)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % 200 == 0:
                dev_step(x_dev, y_dev)

if __name__ == '__main__':
    data = pd.read_csv('../../data/train/processed_datay', sep=',')
    data = data[['label1', 'label2', 'label3']]
    data_x = np.load("../../data/train/processed_datax.npy")
    data["ids"] = data_x

    train_data = data.sample(frac= 0.8).reset_index()
    test = data.sample(frac= 0.2).reset_index()

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
    with open("../../data/label1.txt","r") as f:
        y_label1 = f.read().split(' ')
    with open("../../data/label2.txt","r") as f:
        y_label2 = f.read().split(' ')
    with open("../../data/label3.txt","r") as f:
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

    y_train = [train_y_label1_one_hot,train_y_label2_one_hot,train_y_label3_one_hot]
    y_test = [test_y_label1_one_hot,test_y_label2_one_hot,test_y_label3_one_hot]
    train(train_x, y_train, test_x, y_test)