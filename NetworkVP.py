# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import re
import numpy as np
import tensorflow as tf

from Config import Config
import objgraph

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
from tensorflow.contrib import rnn


#tf.enable_eager_execution()

class NetworkVP:
    def __init__(self, device, model_name, num_actions, is_training):
        #tf.enable_eager_execution()
        self.device = device
        self.model_name = model_name
        self.num_actions = num_actions

        self.img_width = Config.IMAGE_WIDTH
        self.img_height = Config.IMAGE_HEIGHT
        self.img_channels = Config.STACKED_FRAMES

        self.learning_rate = Config.LEARNING_RATE_START
        self.beta = Config.BETA_START
        self.log_epsilon = Config.LOG_EPSILON


        self.Num_Of_Features = 42
        self.BN_EPSILON = 0.001
        self.BN_DECAY = 0.997
        self.MOVING_AVARAGE_DECAY = 0.997
        self.IS_TRAINING = is_training


        self.graph = tf.Graph()
        self.graph2 = tf.Graph()
        self.extractor = tf.load_op_library('./libRadiomicsGLCM.so').radiomics_glcm


        with self.graph.as_default() as g:
            with tf.device(self.device):

                self._create_graph()
                self.sess = tf.Session(
                    graph=self.graph,
                    config=tf.ConfigProto(
                        allow_soft_placement=True,
                        log_device_placement=True,
                        gpu_options=tf.GPUOptions(allow_growth=True)))

                self.sess.run(tf.global_variables_initializer())
                if Config.TENSORBOARD: self._create_tensor_board()
                if Config.LOAD_CHECKPOINT or Config.SAVE_MODELS:
                    vars = tf.global_variables()
                    self.saver = tf.train.Saver({var.name: var for var in vars}, max_to_keep=0)

                #self.sess.graph.finalize()
                ##
                # objgraph.show_refs(self.graph, filename='./graph1.png')
        
        with self.graph2.as_default() as g2:
            with tf.device(self.device):

                self._create_graph2()
                self.sess2 = tf.Session(
                    graph=self.graph2,
                    config=tf.ConfigProto(
                        allow_soft_placement=True,
                        log_device_placement=True,
                        gpu_options=tf.GPUOptions(allow_growth=True)))

                self.sess2.run(tf.global_variables_initializer())
                #self.sess2.graph.finalize()
                #objgraph.show_growth()
                #objgraph.show_refs(self.graph2, filename='./graph2.png')


    def _create_graph2(self):
        self.x1 = tf.placeholder(tf.int32, shape=[None, self.img_height, self.img_width, self.img_channels], name='X1')
        self.features = tf.placeholder(tf.float32, shape=[None, self.img_channels*self.Num_Of_Features], name='features')


        #self.features = tf.add(self.x1, 1)
        # As implemented in A3C paper
        #self.x_to_cuda0 = tf.transpose(self.x1, perm=[0, 3, 1, 2])
        self.x_to_cuda = tf.reshape(tf.transpose(self.x1, perm=[0, 3, 1, 2]), shape=[-1, self.img_height, self.img_width])
        print('extracting' + '\n')
        print('calling GLCM kernel! \n')
        self.radiomics_features0 = self.extractor(self.x_to_cuda)
        print('extracted' + '\n')
        self.radiomics_features1 = tf.reshape(self.radiomics_features0, shape=[self.Num_Of_Features, -1, self.img_channels])
        self.radiomics_features = tf.transpose(self.radiomics_features1, perm=[1, 2, 0])

        #self.original_features = tf.slice(self.radiomics_features, [0, 0, 0], [-1, 20, -1])
        #self.last_features = tf.subtract(tf.slice(self.radiomics_features, [0, 20, 0], [-1, 20, -1]), self.original_features)
        #self.now_features = tf.subtract(tf.slice(self.radiomics_features, [0, 40, 0], [-1, 20, -1]), self.original_features)

        #self.converted_features = tf.concat([self.original_features, self.last_features, self.now_features], axis=0)
        #self.features = tf.reshape(self.converted_features, shape=[-1, self.img_channels * self.Num_Of_Features])
        self.features = tf.reshape(self.radiomics_features, shape=[-1, self.img_channels * self.Num_Of_Features])
        #self.features = tf.reshape(tf.transpose(tf.reshape(self.extractor(self.x_to_cuda), shape=[self.Num_Of_Features, -1, self.img_channels]), perm=[1, 2, 0]), shape=[-1, self.img_channels * self.Num_Of_Features])
        #print('calling finished! \n')

    def _create_graph(self):
        #self.x = tf.placeholder(tf.float32, shape=[None, self.img_height, self.img_width, self.img_channels], name='X1')
        self.input_features = tf.placeholder(tf.float32,
                                             shape=[None, self.Num_Of_Features * self.img_channels], name='input_features')
        #print(str(self.input_features.shape) + '\n')

        self.y_r = tf.placeholder(tf.float32, [None], name='Yr')
        self.var_beta = tf.placeholder(tf.float32, name='beta', shape=[])
        self.var_learning_rate = tf.placeholder(tf.float32, name='lr', shape=[])

        self.global_step = tf.Variable(0, trainable=False, name='step')

        '''
        self.n1 = self.conv2d_layer(self.x, 7, 96, 'conv11', strides=[1, 2, 2, 1])
        self.pn1 = tf.nn.max_pool(self.n1, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')
        self.n2 = self.conv2d_layer(self.pn1, 5, 256, 'conv12', strides=[1, 2, 2, 1])
        self.pn2 = tf.nn.max_pool(self.n2, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')
        self.n3 = self.conv2d_layer(self.pn2, 3, 384, 'conv13', strides=[1, 2, 2, 1])
        self.n4 = self.conv2d_layer(self.n3, 3, 384, 'conv14', strides=[1, 2, 2, 1])
        self.n5 = self.conv2d_layer(self.n4, 3, 256, 'conv15', strides=[1, 2, 2, 1])
        self.pn6 = tf.nn.max_pool(self.n5, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')
        '''
        self.action_index = tf.placeholder(tf.float32, [None, self.num_actions])
        self.batch_size = tf.placeholder(tf.int32, [])

        '''
        _input = self.pn6

        flatten_input_shape = _input.get_shape()
        nb_elements = flatten_input_shape[1] * flatten_input_shape[2] * flatten_input_shape[3]
        self.flat = tf.reshape(_input, shape=[-1, nb_elements._value])
        '''
        #print('****************************************begin bn **********************************************************\n')
        #self.input_LSTM = tf.layers.batch_normalization(self.input_features, training=self.IS_TRAINING)
        #print(str(self.input_LSTM.shape) + '\n')

        #self.LSTM_input = tf.reshape(self.input_features, shape=[-1, 7, int(self.Num_Of_Features * self.img_channels/7)])
        #print('input:' + str(self.LSTM_input.shape))
        #lstm_cell = rnn.GRUCell(num_units=168)
        #keep_prob = tf.placeholder(tf.float32, [])
        #lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0)
        #mlstm_cell = rnn.MultiRNNCell([lstm_cell] * 7, state_is_tuple=True)


        #init_state = mlstm_cell.zero_state(self.batch_size, dtype=tf.float32)


        #self.LSTM_outputs, self.LSTM_state = tf.nn.dynamic_rnn(mlstm_cell, inputs=self.LSTM_input[:, 0:], initial_state=init_state, time_major=False)
        #print(str(self.LSTM_outputs))
        #self.input_of_GA3C = tf.reshape(self.LSTM_outputs, shape=[-1, self.Num_Of_Features * self.img_channels])
        #self.normed_input_of_GA3C = tf.layers.batch_normalization(self.input_of_GA3C, training=self.IS_TRAINING)
        #print('****************************************finish bn *********************************************************\n')

        self.dd1 = self.dense_layer(self.input_features,128, 'dense11')
        self.d1 = self.dense_layer(self.dd1,128, 'dense1')
        #self.d2 = self.dense_layer(self.d1, 32, 'dense2')
        #self.d12 = tf.concat((self.d1, self.d2), axis=1)
        #self.d3 = self.dense_layer(self.d12, 32, 'dense2')

        self.logits_v = tf.squeeze(self.dense_layer(self.d1, 1, 'logits_v', func=None), axis=[1])
        self.cost_v = 0.5 * tf.reduce_sum(tf.square(self.y_r - self.logits_v), axis=0)

        self.logits_p = self.dense_layer(self.d1, self.num_actions, 'logits_p', func=None)

        if Config.USE_LOG_SOFTMAX:
            self.softmax_p = tf.nn.softmax(self.logits_p)
            self.log_softmax_p = tf.nn.log_softmax(self.logits_p)
            self.log_selected_action_prob = tf.reduce_sum(self.log_softmax_p * self.action_index, axis=1)

            self.cost_p_1 = self.log_selected_action_prob * (self.y_r - tf.stop_gradient(self.logits_v))
            self.cost_p_2 = -1 * self.var_beta * \
                        tf.reduce_sum(self.log_softmax_p * self.softmax_p, axis=1)
        else:
            self.softmax_p = (tf.nn.softmax(self.logits_p) + Config.MIN_POLICY) / (1.0 + Config.MIN_POLICY * self.num_actions)
            self.selected_action_prob = tf.reduce_sum(self.softmax_p * self.action_index, axis=1)

            self.cost_p_1 = tf.log(tf.maximum(self.selected_action_prob, self.log_epsilon)) \
                        * (self.y_r - tf.stop_gradient(self.logits_v))
            self.cost_p_2 = -1 * self.var_beta * \
                        tf.reduce_sum(tf.log(tf.maximum(self.softmax_p, self.log_epsilon)) *
                                      self.softmax_p, axis=1)

        self.cost_p_1_agg = tf.reduce_sum(self.cost_p_1, axis=0)
        self.cost_p_2_agg = tf.reduce_sum(self.cost_p_2, axis=0)
        self.cost_p = -(self.cost_p_1_agg + self.cost_p_2_agg)

        if Config.DUAL_RMSPROP:
            self.opt_p = tf.train.RMSPropOptimizer(
                learning_rate=self.var_learning_rate,
                decay=Config.RMSPROP_DECAY,
                momentum=Config.RMSPROP_MOMENTUM,
                epsilon=Config.RMSPROP_EPSILON)

            self.opt_v = tf.train.RMSPropOptimizer(
                learning_rate=self.var_learning_rate,
                decay=Config.RMSPROP_DECAY,
                momentum=Config.RMSPROP_MOMENTUM,
                epsilon=Config.RMSPROP_EPSILON)
        else:
            self.cost_all = self.cost_p + self.cost_v
            self.opt = tf.train.RMSPropOptimizer(
                learning_rate=self.var_learning_rate,
                decay=Config.RMSPROP_DECAY,
                momentum=Config.RMSPROP_MOMENTUM,
                epsilon=Config.RMSPROP_EPSILON)

        if Config.USE_GRAD_CLIP:
            if Config.DUAL_RMSPROP:
                self.opt_grad_v = self.opt_v.compute_gradients(self.cost_v)
                self.opt_grad_v_clipped = [(tf.clip_by_norm(g, Config.GRAD_CLIP_NORM),v)
                                            for g,v in self.opt_grad_v if not g is None]
                self.train_op_v = self.opt_v.apply_gradients(self.opt_grad_v_clipped)

                self.opt_grad_p = self.opt_p.compute_gradients(self.cost_p)
                self.opt_grad_p_clipped = [(tf.clip_by_norm(g, Config.GRAD_CLIP_NORM),v)
                                            for g,v in self.opt_grad_p if not g is None]
                self.train_op_p = self.opt_p.apply_gradients(self.opt_grad_p_clipped)
                self.train_op = [self.train_op_p, self.train_op_v]
            else:
                self.opt_grad = self.opt.compute_gradients(self.cost_all)
                self.opt_grad_clipped = [(tf.clip_by_average_norm(g, Config.GRAD_CLIP_NORM),v) for g,v in self.opt_grad]
                self.train_op = self.opt.apply_gradients(self.opt_grad_clipped)
        else:
            if Config.DUAL_RMSPROP:
                self.train_op_v = self.opt_p.minimize(self.cost_v, global_step=self.global_step)
                self.train_op_p = self.opt_v.minimize(self.cost_p, global_step=self.global_step)
                self.train_op = [self.train_op_p, self.train_op_v]
            else:
                self.train_op = self.opt.minimize(self.cost_all, global_step=self.global_step)




    def _create_tensor_board(self):
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        summaries.append(tf.summary.scalar("Pcost_advantage", self.cost_p_1_agg))
        summaries.append(tf.summary.scalar("Pcost_entropy", self.cost_p_2_agg))
        summaries.append(tf.summary.scalar("Pcost", self.cost_p))
        summaries.append(tf.summary.scalar("Vcost", self.cost_v))
        summaries.append(tf.summary.scalar("LearningRate", self.var_learning_rate))
        summaries.append(tf.summary.scalar("Beta", self.var_beta))
        #for var in tf.trainable_variables():
        #    summaries.append(tf.summary.histogram("weights_%s" % var.name, var))

        #summaries.append(tf.summary.histogram("features", self.input_features))
        #summaries.append(tf.summary.histogram("activation_n2", self.n2))
        #summaries.append(tf.summary.histogram("activation_d2", self.d1))
        #summaries.append(tf.summary.histogram("activation_v", self.logits_v))
        #summaries.append(tf.summary.histogram("activation_p", self.softmax_p))

        self.summary_op = tf.summary.merge(summaries)
        #self.log_writer = tf.summary.FileWriter("logs/%s" % self.model_name, self.sess2.graph)
        self.log_writer = tf.summary.FileWriter("logs/%s" % self.model_name, self.sess.graph)

    def dense_layer(self, input, out_dim, name, func=tf.nn.relu):
        in_dim = input.get_shape().as_list()[-1]
        d = 1.0 / np.sqrt(in_dim)
        with tf.variable_scope(name):
            w_init = tf.random_uniform_initializer(-d, d)
            b_init = tf.random_uniform_initializer(-d, d)
            w = tf.get_variable('w', dtype=tf.float32, shape=[in_dim, out_dim], initializer=w_init)
            b = tf.get_variable('b', shape=[out_dim], initializer=b_init)
            #print(str(input) + '\n')
            output = tf.matmul(input, w) + b
            out = tf.layers.batch_normalization(output, training=self.IS_TRAINING)
            if func is not None:
                output = func(output)

        return out

    def conv2d_layer(self, input, filter_size, out_dim, name, strides, func=tf.nn.relu):
        in_dim = input.get_shape().as_list()[-1]
        d = 1.0 / np.sqrt(filter_size * filter_size * in_dim)
        with tf.variable_scope(name):
            w_init = tf.random_uniform_initializer(-d, d)
            b_init = tf.random_uniform_initializer(-d, d)
            w = tf.get_variable('w',
                                shape=[filter_size, filter_size, in_dim, out_dim],
                                dtype=tf.float32,
                                initializer=w_init)
            b = tf.get_variable('b', shape=[out_dim], initializer=b_init)

            output = tf.nn.conv2d(input, w, strides=strides, padding='SAME') + b
            if func is not None:
                output = func(output)

        return output

    '''
    def bn(self):

        #update_moving_mean = moving_averages.assign_moving_average(self.moving_mean, self.mean, self.BN_DECAY)
        #update_moving_variance = moving_averages.assign_moving_average(self.moving_variance, self.variance, self.BN_DECAY)
        #tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
        #tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

        self.mean, self.variance = tf.cond(
            self.tf_is_training, lambda: (self.mean, self.variance),
            lambda: (self.moving_mean, self.moving_variance))

        normalized = tf.nn.batch_normalization(self.input_features, self.mean, self.variance, self.beta, self.gamma, self.BN_EPSILON)
        #print('normalized!')
        return normalized
    '''



    def __get_base_feed_dict(self):
        return {self.var_beta: self.beta, self.var_learning_rate: self.learning_rate}

    def get_global_step(self):
        step = self.sess.run(self.global_step)
        #step = self.global_step
        return step

    def predict_single(self, x):
        self.IS_TRAINING = False
        return self.predict_p(x[None, :])[0]

    def predict_v(self, x):
        self.IS_TRAINING = False
        features = self.sess2.run(self.features, feed_dict={self.x1: x})
        batch_size = features.shape[0]
        #objgraph.show_growth()
        prediction = self.sess.run(self.logits_v, feed_dict={self.input_features: features, self.batch_size: batch_size})
        return prediction

    def predict_p(self, x):
        self.IS_TRAINING = False
        features = self.sess2.run(self.features, feed_dict={self.x1: x})
        batch_size = features.shape[0]
        #objgraph.show_growth()
        #objgraph.show_refs(self.graph, filename='./grap1.png')
        prediction = self.sess.run(self.softmax_p, feed_dict={self.input_features: features, self.batch_size: batch_size})
        return prediction

    def predict_p_and_v(self, x):
        self.IS_TRAINING = False
        features = self.sess2.run(self.features, feed_dict={self.x1: x})
        batch_size = features.shape[0]

        #objgraph.show_growth()
        #objgraph.show_refs(self.graph, filename='./graph1.png')
        #print('shape:' + str(x.shape))
        return self.sess.run([self.softmax_p, self.logits_v], feed_dict={self.input_features: features, self.batch_size: batch_size})

    def train(self, x, y_r, a, trainer_id):
        self.IS_TRAINING = True
        feed_dict = self.__get_base_feed_dict()
        #tf.reset_default_graph()
        #self.sess.reset(self.features)
        #print('net_input_shape:' + str(x.shape))
        features = self.sess2.run(self.features, feed_dict={self.x1: x})
        batch_size = features.shape[0]
        #objgraph.show_growth()
        #objgraph.show_refs(self.graph, filename='./graph1.png')
        feed_dict.update({self.input_features: features, self.batch_size: batch_size, self.y_r: y_r, self.action_index: a})
        self.sess.run(self.train_op, feed_dict=feed_dict)

    def log(self, x, y_r, a):
        self.IS_TRAINING = False
        feed_dict = self.__get_base_feed_dict()

        features = self.sess2.run(self.features, feed_dict={self.x1: x})
        #tf.reset_default_graph()
        #objgraph.show_growth()
        #objgraph.show_refs(self.features, filename='./graph2.png')
        batch_size = features.shape[0]
        feed_dict.update({self.input_features: features, self.batch_size: batch_size, self.y_r: y_r, self.action_index: a})
        step, summary = self.sess.run([self.global_step, self.summary_op], feed_dict=feed_dict)
        self.log_writer.add_summary(summary, step)

    def _checkpoint_filename(self, episode):
        return './checkpoints/%s_%08d' % (self.model_name, episode)

    def _get_episode_from_filename(self, filename):
        # TODO: hacky way of getting the episode. ideally episode should be stored as a TF variable
        return int(re.split('/|/|_', filename)[3])

    def save(self, episode):
        self.saver.save(self.sess, self._checkpoint_filename(episode))

    def load(self):
        filename = tf.train.latest_checkpoint(os.path.dirname(self._checkpoint_filename(episode=1000)))
        if Config.LOAD_EPISODE > 0:
            filename = self._checkpoint_filename(Config.LOAD_EPISODE)
        self.saver.restore(self.sess, filename)
        return self._get_episode_from_filename(filename)

    def get_variables_names(self):
        return [var.name for var in self.graph.get_collection('trainable_variables')]

    def get_variable_value(self, name):
        return self.sess.run(self.graph.get_tensor_by_name(name))


    def extract(self, x):
        features = self.sess2.run(self.features, feed_dict={self.x1:x})
        objgraph.show_growth()
        #self.sess2._default_graph_context_manager()
        #self.sess2._default_session_context_manager()
        #self.sess2.reset(self.sess2.graph)
        #self.sess2.reset(self.x)
        #self.sess2.as_default()
        return features