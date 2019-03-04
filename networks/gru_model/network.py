import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import tensorflow.contrib.layers as layers
import tensorflow as tf
import numpy as np


import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import tensorflow.contrib.layers as layers
class Gru(object):
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
        
        #gru模型
        with tf.variable_scope('bigru_text'):
            self.gru_output = self.bigru_inference(self.embedded_chars)
            print("gru shape", self.gru_output.shape)
            
        self.h_drop = self.gru_output
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
        
        self.saver = tf.train.Saver(max_to_keep = 2)
    def gru_cell(self):
        with tf.name_scope('gru_cell'):
            cell = rnn.GRUCell(self.hidden_size, reuse=tf.get_variable_scope().reuse)
        return rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

    def bi_gru(self, inputs):
        """build the bi-GRU network. 返回个所有层的隐含状态。"""
        cells_fw = [self.gru_cell() for _ in range(self.n_layer)]
        cells_bw = [self.gru_cell() for _ in range(self.n_layer)]
        initial_states_fw = [cell_fw.zero_state(self.batch_size, tf.float32) for cell_fw in cells_fw]
        initial_states_bw = [cell_bw.zero_state(self.batch_size, tf.float32) for cell_bw in cells_bw]
        outputs, _, _ = rnn.stack_bidirectional_dynamic_rnn(cells_fw, cells_bw, inputs,
                                                            initial_states_fw=initial_states_fw,
                                                            initial_states_bw=initial_states_bw, dtype=tf.float32)
        return outputs
    
    def bigru_inference(self, X_inputs):
#         inputs = tf.nn.embedding_lookup(self.title_embedding, X_inputs)
        output_bigru = self.bi_gru(X_inputs)
        output_att = self.task_specific_attention(output_bigru, self.hidden_size*2)
        return output_att
    
    def task_specific_attention(self, inputs, output_size,
                                initializer=layers.xavier_initializer(),
                                activation_fn=tf.tanh, scope=None):
        """
        Performs task-specific attention reduction, using learned
        attention context vector (constant within task of interest).
        Args:
            inputs: Tensor of shape [batch_size, units, input_size]
                `input_size` must be static (known)
                `units` axis will be attended over (reduced from output)
                `batch_size` will be preserved
            output_size: Size of output's inner (feature) dimension
        Returns:
           outputs: Tensor of shape [batch_size, output_dim].
        """
#         assert len(inputs.get_shape()) == 3 and inputs.get_shape()[-1].value is not None
        with tf.variable_scope(scope or 'attention') as scope:
            # u_w, attention 向量
            attention_context_vector = tf.get_variable(name='attention_context_vector', shape=[output_size],
                                                       initializer=initializer, dtype=tf.float32)
            # 全连接层，把 h_i 转为 u_i ， shape= [batch_size, units, input_size] -> [batch_size, units, output_size]
            input_projection = layers.fully_connected(inputs, output_size, activation_fn=activation_fn, scope=scope)
            # 输出 [batch_size, units]
            vector_attn = tf.reduce_sum(tf.multiply(input_projection, attention_context_vector), axis=2, keep_dims=True)
            attention_weights = tf.nn.softmax(vector_attn, dim=1)
            tf.summary.histogram('attention_weigths', attention_weights)
            weighted_projection = tf.multiply(inputs, attention_weights)
            outputs = tf.reduce_sum(weighted_projection, axis=1)
            return outputs  # 输出 [batch_size, hidden_size*2]