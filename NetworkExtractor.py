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


# tf.enable_eager_execution()

class NetworkExtractor:
    def __init__(self, device, model_name, num_actions, is_training):
        # tf.enable_eager_execution()
        self.device = device
        self.model_name = model_name
        self.num_actions = num_actions

        self.img_width = Config.IMAGE_WIDTH
        self.img_height = Config.IMAGE_HEIGHT
        self.img_channels = Config.STACKED_FRAMES

        self.learning_rate = Config.LEARNING_RATE_START
        self.beta = Config.BETA_START
        self.log_epsilon = Config.LOG_EPSILON

        self.Num_Of_Features = 23
        self.BN_EPSILON = 0.001
        self.BN_DECAY = 0.9997
        self.MOVING_AVARAGE_DECAY = 0.9997
        self.IS_TRAINING = is_training

        self.graph = tf.Graph()
        self.graph2 = tf.Graph()
        self.extractor = tf.load_op_library('./libRadiomicsGLCM.so').radiomics_glcm

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
                # self.sess2.graph.finalize()
                # objgraph.show_growth()
                # objgraph.show_refs(self.graph2, filename='./graph2.png')

    def _create_graph2(self):
        self.x1 = tf.placeholder(tf.int32, shape=[None, self.img_height, self.img_width], name='X1')
        self.features = tf.placeholder(tf.float32, shape=[None, self.img_channels * self.Num_Of_Features],
                                       name='features')
        self.features = self.extractor(self.x1)


    def extract(self, x):
        features = self.sess2.run(self.features, feed_dict={self.x1: x})
        return features