import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import tensorflow.contrib.layers as layers
class InceptionModel(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self, 
                 sequence_length, 
                 num_classes, 
                 embedding_size,  
                 num_filters, 
                 l2_reg_lambda=0.0, 
                 n_layer = 1, 
                 hidden_size = 32, 
                 batch_size = 256, 
                 vac_size = 27440,
                 is_training = True):
        """
        sequence_length : 一个句子的长度（词的个数）
        embedding_size : 词向量的长度
        num_classes : 三个标签的类别数
        vac_size : 词的个数
        """
        # Placeholders for input, output and dropout

        self.n_layer = n_layer
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_filters = num_filters
        with tf.name_scope("Input"):
            self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
            self.input_y1 = tf.placeholder(tf.float32, [None, num_classes[0]], name="input_y1")
            self.input_y2 = tf.placeholder(tf.float32, [None, num_classes[1]], name="input_y2")
            self.input_y3 = tf.placeholder(tf.float32, [None, num_classes[2]], name="input_y3")
            self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")


        
        with tf.name_scope("embedding"):
            W = tf.Variable(
                tf.random_uniform([274420, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)

        l2_loss = tf.constant(0.0)
        
        self.embedded_input = tf.layers.batch_normalization(self.embedded_chars, training=is_training)
        self.embedded_input = tf.expand_dims(self.embedded_input, -1)
        
        with tf.variable_scope("cnn1"):
            filter_shape = [3, 3, 1, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
            self.conv_output = tf.nn.conv2d(
                    self.embedded_input,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="SAME",
                    name="conv")
            self.conv_output = tf.nn.dropout(self.conv_output, self.keep_prob)
            
        #gru模型
        with tf.variable_scope('inception_text'):
            self.output = self.text_inception(self.conv_output)
        
        with tf.variable_scope("cnn2"):
            filter_shape = [5, 5, 16, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
            self.conv_output = tf.nn.conv2d(
                    self.output,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
            self.conv_output = tf.nn.dropout(self.conv_output, self.keep_prob)
        self.h_drop = tf.reshape(self.conv_output,(-1, 16 * 20 * 4))
        num_filters_total = self.h_drop.shape[1]
        
        
        with tf.name_scope("output1"):
            W1 = tf.get_variable(
                "W11",
                shape=[num_filters_total, 64],
                initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.Variable(tf.constant(0.1, shape=[64]), name="b11")
            self.output = tf.nn.xw_plus_b(self.h_drop, W1, b1, name="output1")
            self.output = tf.layers.batch_normalization(self.output, training=is_training)
            self.output = tf.nn.relu(self.output)
            self.output = tf.nn.dropout(self.output, self.keep_prob)
            W2 = tf.get_variable(
                "W12",
                shape=[64, num_classes[0]],
                initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.Variable(tf.constant(0.1, shape=[num_classes[0]]), name="b12")
            l2_loss += tf.nn.l2_loss(W1)
            l2_loss += tf.nn.l2_loss(b1)
            self.scores1 = tf.nn.xw_plus_b(self.output, W2, b2, name="scores")
            self.predictions1 = tf.argmax(self.scores1, 1, name="predictions")
            
        with tf.name_scope("output2"):
            W1 = tf.get_variable(
                "W21",
                shape=[num_filters_total, 128],
                initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.Variable(tf.constant(0.1, shape=[128]), name="b21")
            self.output = tf.nn.xw_plus_b(self.h_drop, W1, b1, name="output2")
            self.output = tf.layers.batch_normalization(self.output, training=is_training)
            self.output = tf.nn.relu(self.output)
            self.output = tf.nn.dropout(self.output, self.keep_prob)
            W2 = tf.get_variable(
                "W22",
                shape=[128, num_classes[1]],
                initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.Variable(tf.constant(0.1, shape=[num_classes[1]]), name="b22")
            l2_loss += tf.nn.l2_loss(W1)
            l2_loss += tf.nn.l2_loss(b1)
            self.scores2 = tf.nn.xw_plus_b(self.output, W2, b2, name="scores")
            self.predictions2 = tf.argmax(self.scores2, 1, name="predictions")
            
        with tf.name_scope("output3"):
            W1 = tf.get_variable(
                "W31",
                shape=[num_filters_total, 256],
                initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.Variable(tf.constant(0.1, shape=[256]), name="b31")
            self.output = tf.nn.xw_plus_b(self.h_drop, W1, b1, name="output3")
            self.output = tf.layers.batch_normalization(self.output, training=is_training)
            self.output = tf.nn.relu(self.output)
            self.output = tf.nn.dropout(self.output, self.keep_prob)
            W2 = tf.get_variable(
                "W32",
                shape=[256, num_classes[2]],
                initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.Variable(tf.constant(0.1, shape=[num_classes[2]]), name="b32")
            l2_loss += tf.nn.l2_loss(W1)
            l2_loss += tf.nn.l2_loss(b1)
            self.scores3 = tf.nn.xw_plus_b(self.output, W2, b2, name="scores")
            self.predictions3 = tf.argmax(self.scores3, 1, name="predictions")
        # Calculate mean cross-entropy loss
        
        with tf.name_scope("loss"):
            losses1 = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores1, labels=self.input_y1)
            losses2 = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores2, labels=self.input_y2)
            losses3 = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores3, labels=self.input_y3)
            self.loss = 0.1 * tf.reduce_mean(losses1) + 0.2 * tf.reduce_mean(losses2) + 0.7*tf.reduce_mean(losses3) + l2_reg_lambda * l2_loss
            tf.summary.scalar('loss1',tf.reduce_mean(losses1))
            tf.summary.scalar('loss2',tf.reduce_mean(losses2))
            tf.summary.scalar('loss3',tf.reduce_mean(losses3))
            tf.summary.scalar('loss',self.loss)
        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions1 = tf.equal(self.predictions1, tf.argmax(self.input_y1, 1))
            correct_predictions2 = tf.equal(self.predictions2, tf.argmax(self.input_y2, 1))
            correct_predictions3 = tf.equal(self.predictions3, tf.argmax(self.input_y3, 1))
#           单独的准确率
            self.accuracy1 = tf.reduce_mean(tf.cast(correct_predictions1, "float"), name="accuracy1")
            self.accuracy2 = tf.reduce_mean(tf.cast(correct_predictions2, "float"), name="accuracy2")
            self.accuracy3 = tf.reduce_mean(tf.cast(correct_predictions3, "float"), name="accuracy3")
#           一起的准确率
            self.acc = tf.reduce_mean(tf.cast(correct_predictions1, "float")*tf.cast(correct_predictions2, "float")*tf.cast(correct_predictions3, "float"))
            tf.summary.scalar('acc1',self.accuracy1)
            tf.summary.scalar('acc2',self.accuracy2)
            tf.summary.scalar('acc3',self.accuracy3)
            tf.summary.scalar('acc',self.acc)

        self.saver = tf.train.Saver(max_to_keep=2)


    def text_inception(self, inputs):
        output = []
        with tf.name_scope("conv-1"):
            filter_shape = [1, 1, self.num_filters, self.num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
            conv = tf.nn.conv2d(
                    inputs,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
            conv = tf.nn.dropout(conv, self.keep_prob)
            output.append(conv)
        with tf.name_scope("conv-2"):
            filter_shape1 = [1, 1, self.num_filters, self.num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape1, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
            conv1 = tf.nn.conv2d(
                    inputs,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
            filter_shape2 = [3, 3, self.num_filters, 2 * self.num_filters]
            W1 = tf.Variable(tf.truncated_normal(filter_shape2, stddev=0.1), name="W")
            b1 = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
            conv2 = tf.nn.conv2d(
                    conv1,
                    W1,
                    strides=[1, 1, 1, 1],
                    padding="SAME",
                    name="conv")
            conv2 = tf.nn.dropout(conv2, self.keep_prob)
            output.append(conv2)
        with tf.name_scope("conv-4"):
            filter_shape = [3, 3, self.num_filters, self.num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
            conv = tf.nn.conv2d(
                    inputs,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="SAME",
                    name="conv")
            conv = tf.nn.dropout(conv, self.keep_prob)
            output.append(conv)
        
        return tf.concat(output, 3)