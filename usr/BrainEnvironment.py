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

import sys
if sys.version_info >= (3,0):
    from queue import Queue
else:
    from Queue import Queue

import numpy as np
import scipy.misc as misc
import gym
from gym import spaces
import cv2
from gym.envs.usr.image_helper import *
from gym.envs.usr.metrics import *
from gym.envs.usr.parse_xml_annotations import *
from gym.envs.usr.reinforcement import *
#from gym.envs.usr.extractor_net import GLCM
#tf.enable_eager_execution()
from gym.envs.usr.NetworkExtractor import NetworkExtractor

class Environment(gym.Env):
    def __init__(self, Config, path="/home/jyn/GA3C/data/BRATS/", DataSetSelect='ALL', SEG='0', frames=28, height=240, weight=240):


        # get path of the dataset
        self.path = path
        self.DataSetSelect = DataSetSelect
        self.SEG = SEG

        # set path of images and masks
        self.img_path = self.path + self.DataSetSelect + "/"  # t1ce/"
        self.seg_path = self.path + self.DataSetSelect + "/t1ce/SegmentationClass" + SEG + '/'

        # set path of image and mask of different modality
        self.img_path_t1ce = self.img_path + "t1ce/"
        self.img_path_flair = self.img_path + "flair/"
        self.img_path_t1 = self.img_path + "t1/"
        self.img_path_t2 = self.img_path + "t2/"

        # get image names
        if Config.PLAY_MODE == False:
            CHOSEN = 'cv0_' + 'trainval'
        else:
            CHOSEN = 'cv0_' + 'test'
        self.image_names = np.array([load_images_names_in_data_set(CHOSEN, self.img_path_t1ce)])
        self.num_of_image_per_m = len(self.image_names[0])


        self.path_model = '../model_image_zooms/'
        self.alpha = 0.2
        # initalization
        self.nb_frames = frames
        self.frame_q = Queue(maxsize=self.nb_frames)

        self.frame_now = None
        self.feature_now = None
        self.iou_now = None
        self.p_mask_now = None

        self.previous_state = None
        self.current_state = None
        self.total_reward = 0

        self.offset = (0, 0)
        self.action_space = spaces.Discrete(4)


        self.height = height
        self.weight = weight
        self.init_height = 240
        self.init_weight = 240
        #self.extractor = tf.load_op_library('./libRadiomicsGLCM.so').radiomics_glcm

        self.reset()

    def get_num_actions(self):
        return int(4)

    def preprocess(self, region_image):
        return np.round(region_image).astype('int32')


    def _get_current_state(self):
        if not self.frame_q.full():
            return None  # frame queue is not full yet.
        x_ = np.array(self.frame_q.queue)
        x_ = np.transpose(x_, [1, 2, 0])  # move channels
        #features_ = self.model.extract(x_)
        return x_

    def _update_frame_q(self, region_image):

        # update cropped images
        self.frame_q.queue.clear()
        self.frame_now = self.preprocess(region_image)


        # if next action == 0

        # Left top
        self.offset0 = (self.offset[0], self.offset[1])
        self.size0 = (self.size[0] - self.alpha_w, self.size[1] - self.alpha_h)
        self.frame_next_0 = self.preprocess(update_image(self.initial_image, self.offset0, self.size0))

        # if next action == 1
        # Left bottom
        self.offset1 = (self.offset[0] + self.alpha_w, self.offset[1])
        self.size1 = (self.size[0] - self.alpha_w, self.size[1] - self.alpha_h)
        self.frame_next_1 = self.preprocess(update_image(self.initial_image, self.offset1, self.size1))

        # if next action == 2
        # Right top
        self.offset2 = (self.offset[0], self.offset[1] + self.alpha_h)
        self.size2 = (self.size[0] - self.alpha_w, self.size[1] - self.alpha_h)
        self.frame_next_2 = self.preprocess(update_image(self.initial_image, self.offset2, self.size2))

        # if next action == 3
        # Right bottom
        self.offset3 = (self.offset[0] + self.alpha_w, self.offset[1] + self.alpha_h)
        self.size3 = (self.size[0] - self.alpha_w, self.size[1] - self.alpha_h)
        self.frame_next_3 = self.preprocess(update_image(self.initial_image, self.offset3, self.size3))

        '''
        self.offset_next = {}
        self.frame_next = {}
        A = 16
        for i in range(A):
                self.offset_next[i] = (int(i/4) * int(self.size[0]/4), np.remainder(i, 4) * int(self.size[1]/4))
        for i in range(A):
            self.frame_next[i] = self.preprocess(update_image(self.initial_image, self.offset_next[i], (int(self.size[0]/4),
                                                                                                   int(self.size[1]/4))))
        '''
        '''
        # if next action == 4
        # center
        self.offset4 = (self.offset[0] + self.alpha_w / 2, self.offset[1] + self.alpha_h / 2)
        self.size4 = (self.size[0] - self.alpha_w, self.size[1] - self.alpha_h)
        self.frame_next_4 = self.preprocess(update_image(self.initial_image, self.offset4, self.size4))
        '''


        if self.current_state != np.array([]):


            for i in range(region_image.shape[2]):
                self.frame_q.put(self.frame_next_0[:, :, i].astype('int32'))
            for i in range(region_image.shape[2]):
                self.frame_q.put(self.frame_next_1[:, :, i].astype('int32'))
            for i in range(region_image.shape[2]):
                self.frame_q.put(self.frame_next_2[:, :, i].astype('int32'))
            for i in range(region_image.shape[2]):
                self.frame_q.put(self.frame_next_3[:, :, i].astype('int32'))


            #for j in range(A):
            #    for i in range(region_image.shape[2]):
            #        self.frame_q.put(self.frame_next[j][:, :, i].astype('int32'))

            for i in range(region_image.shape[2]):
                self.frame_q.put(self.frame_now[:, :, i].astype('int32'))

            for i in range(region_image.shape[2]):
                self.frame_q.put(self.current_state[:, :, i].astype('int32'))


            #for i in range(region_image.shape[2]):
            #    self.frame_q.put(self.previous_state[:, :, i].astype('int32'))


            for i in range(self.initial_image.shape[2]):
                self.frame_q.put(self.initial_image[:, :, i].astype('int32'))

        else:


            for i in range(region_image.shape[2]):
                self.frame_q.put(self.frame_next_0[:, :, i].astype('int32'))
            for i in range(region_image.shape[2]):
                self.frame_q.put(self.frame_next_1[:, :, i].astype('int32'))
            for i in range(region_image.shape[2]):
                self.frame_q.put(self.frame_next_2[:, :, i].astype('int32'))
            for i in range(region_image.shape[2]):
                self.frame_q.put(self.frame_next_3[:, :, i].astype('int32'))



            #for j in range(A):
            #    for i in range(region_image.shape[2]):
            #        self.frame_q.put(self.frame_next[j][:, :, i].astype('int32'))
            for i in range(region_image.shape[2]):
                self.frame_q.put(self.frame_now[:, :, i].astype('int32'))



            for i in range(self.initial_image.shape[2]):
                self.frame_q.put(self.initial_image[:, :, i].astype('int32'))

            for i in range(self.frame_now.shape[2]):
                self.frame_q.put(self.initial_image[:, :, i].astype('int32'))


        #print('shape: ' + str(self.current_state.shape) + '\n')
        # update feature
        '''
        radiomics_descriptor = \
            get_radiomics_descriptor_for_image(image, yaml='./Params.yaml')
        self.feature_q.put(radiomics_descriptor)
        '''

    def reset(self):
        '''
        self.total_reward = 0
        self.frame_q.queue.clear()
        self._update_frame_q(self.game.reset())
        self.previous_state = self.current_state = None
        '''
        #print('reseting \n')
        # clearning queues
        self.total_reward = 0
        self.frame_q.queue.clear()


        # reading images

        id = np.random.randint(0, self.num_of_image_per_m, 1)
        self.img_name = self.image_names[0][id][0][:-1]
        t1ce_img_name = self.img_path_t1ce + '/JPEGImages/' + self.img_name + '.png'

        flair_img_name = self.img_path_flair + '/JPEGImages/' + self.img_name + '.png'
        t1_img_name = self.img_path_t1 + '/JPEGImages/' + self.img_name + '.png'
        t2_img_name = self.img_path_t2 + '/JPEGImages/' + self.img_name + '.png'

        #print(str(t1ce_img_name) + '\n' + str(flair_img_name) + '\n' + str(t1_img_name) + '\n' + str(t2_img_name) + '\n')

        img_t1ce = cv2.imread(t1ce_img_name)

        img_flair = cv2.imread(flair_img_name)
        img_t1 = cv2.imread(t1_img_name)
        img_t2 = cv2.imread(t2_img_name)


        # initializng masks and shapes
        self.region_image = np.zeros((img_t1ce.shape[0], img_t1ce.shape[1], 4))

        '''
        wavelet_t1ce = getWaveletImage(img_t1ce[:, :, 0].astype(np.float32))
        #print('size:' + str(wavelet_t1ce.shape) + str(self.height) + str(self.weight) + '\n')
        
        wavelet_flair = getWaveletImage(img_flair[:, :, 0].astype(np.float32))
        wavelet_t1 = getWaveletImage(img_t1[:, :, 0].astype(np.float32))
        wavelet_t2 = getWaveletImage(img_t2[:, :, 0].astype(np.float32))
        '''

        '''
        self.region_image[:, :, np.arange(5)] = np.dstack((img_t1ce[:, :, 0], wavelet_t1ce))
        
        self.region_image[:, :, np.arange(5) + 5] = np.dstack((img_flair[:, :, 0], wavelet_flair))
        self.region_image[:, :, np.arange(5) + 10] = np.dstack((img_t1[:, :, 0], wavelet_t1))
        self.region_image[:, :, np.arange(5) + 15] = np.dstack((img_t2[:, :, 0], wavelet_t2))
        '''


        self.region_image[:, :, np.arange(1)] = img_t1ce[:, :, 0][:, :, None]
        self.region_image[:, :, np.arange(1) + 1] = img_flair[:, :, 0][:, :, None]
        self.region_image[:, :, np.arange(1) + 2] = img_t1[:, :, 0][:, :, None]
        self.region_image[:, :, np.arange(1) + 3] = img_t2[:, :, 0][:, :, None]



        self.initial_image = self.preprocess(self.region_image)

        # mask initialization
        self.region_now = np.zeros((self.region_image.shape[0], self.region_image.shape[1]))

        pos_xmin = int(self.initial_image.shape[0]/2 - self.init_weight/2)
        pos_ymin = int(self.initial_image.shape[1]/2 - self.init_height/2)
        pos_xmax = int(pos_xmin + self.init_weight)
        pos_ymax = int(pos_ymin + self.init_height)
        
        self.region_now[pos_xmin:pos_xmax, pos_ymin:pos_ymax] = 1

        # region image initialization
        self.original_shape = (img_t1ce.shape[0], img_t1ce.shape[1])
        self.size = (self.init_height, self.init_weight)
        self.offset = (pos_xmin, pos_ymin)

        self.region_image = update_image(self.initial_image, self.offset, self.size)

        # set par for net step
        self.alpha_w = int(self.alpha * self.size[0])
        self.alpha_h = int(self.alpha * self.size[1])

        # iou computation
        self.gt_mask = generate_bounding_box_from_segmentation(self.seg_path, self.img_name)
        #print('gt: ' + str(self.gt_mask.shape) + ', region_now: ' + str(self.region_now.shape) + '\n')
        self.iou_now, self.p_mask_now = follow_iou(self.gt_mask, self.region_now)
        self.initial_p_mask = self.p_mask_now
        '''
        features = get_radiomics_descriptor_for_image(image, yaml='./Params.yaml')
        self._update_frame_q(features)
        '''
        # get state
        self.previous_state = self.current_state = np.array([])
        self._update_frame_q(self.region_image)
        #self.previous_features = self.current_features = None


    def step(self, action):
        #print('step:' + str(action) + '\n')

        if action == 4:
            done = True
            self.iou_now, self.p_mask_now = follow_iou(self.gt_mask, self.region_now)
            reward = get_reward_trigger(self.iou_now, self.p_mask_now, self.initial_p_mask)
            self._update_frame_q(self.frame_now)

            '''
            # recording
            if 'maxQ' not in vars().keys():
                maxQ = 0
            print('TRAIN: ' + str(TRAIN_COUNT) + ', reward: ' + str(reward) + ', maxQ: ' + str(maxQ) + ', IOU: ' + str(
                new_iou))

            step += 1
            '''
        # movement action, we perform the crop of the corresponding subregion
        else:

            done = False
            self.region_now = np.zeros(self.original_shape)
            # size_mask = (size_mask[0] * scale_subregion, size_mask[1] * scale_subregion)

            if action == 0:
                # Left top
                self.offset = (self.offset[0], self.offset[1])
                self.size = (self.size[0] - self.alpha_w, self.size[1] - self.alpha_h)
            elif action == 1:
                # Left bottom
                self.offset = (self.offset[0] + self.alpha_w, self.offset[1])
                self.size = (self.size[0] - self.alpha_w, self.size[1] - self.alpha_h)
                # offset_aux = (0, size_mask[1] * scale_mask)
                # offset = (offset[0], offset[1] + size_mask[1] * scale_mask)
            elif action == 2:
                # Right top
                self.offset = (self.offset[0], self.offset[1] + self.alpha_h)
                self.size = (self.size[0] - self.alpha_w, self.size[1] - self.alpha_h)
                # offset_aux = (size_mask[0] * scale_mask, 0)
                # offset = (offset[0] + size_mask[0] * scale_mask, offset[1])
            elif action == 3:
                # Right bottom
                self.offset = (self.offset[0] + self.alpha_w, self.offset[1] + self.alpha_h)
                self.size = (self.size[0] - self.alpha_w, self.size[1] - self.alpha_h)
            '''

            done = False
            self.region_now = np.zeros(self.original_shape)
            # size_mask = (size_mask[0] * scale_subregion, size_mask[1] * scale_subregion)
            if action == 0:
                # RIGHT
                offset_aux = (0, self.alpha_h)
                self.offset = (self.offset[0], np.min((self.offset[1] + self.alpha_h, self.original_shape[1] - 1)))
                self.size = (self.size[0], np.min((self.size[1], self.original_shape[1] - self.offset[1] - 1)))
            elif action == 1:
                # LEFT
                offset_aux = (0, -self.alpha_h)
                self.offset = (self.offset[0], np.max((self.offset[1] - self.alpha_h, 0)))
                # offset_aux = (0, size_mask[1] * scale_mask)
                # offset = (offset[0], offset[1] + size_mask[1] * scale_mask)
            elif action == 2:
                # UP
                offset_aux = (-self.alpha_w, 0)
                self.offset = (np.max((self.offset[0] - self.alpha_w, 0)), self.offset[1])
                # offset_aux = (size_mask[0] * scale_mask, 0)
                # offset = (offset[0] + size_mask[0] * scale_mask, offset[1])
            elif action == 3:
                # DOWN
                offset_aux = (self.alpha_w, 0)
                self.offset = (np.min((self.offset[0] + self.alpha_w, self.original_shape[0] - 1)), self.offset[1])
                self.size = (np.min((self.size[0], self.original_shape[0] - self.offset[0] - 1)), self.size[1])
                # offset_aux = (size_mask[0] * scale_mask,
                #              size_mask[1] * scale_mask)
                # offset = (offset[0] + size_mask[0] * scale_mask,
                #          offset[1] + size_mask[1] * scale_mask)
            elif action == 4:
                # BIGGER
                self.size = (np.min((self.original_shape[0] - self.offset[0] - 1, self.size[0] + self.alpha_w)),
                             np.min((self.original_shape[1] - self.offset[1] - 1, self.size[1] + self.alpha_h)))

            elif action == 5:
                # SMALLER
                self.size = (self.size[0] - self.alpha_w, self.size[1] - self.alpha_h)

                # offset_aux = (size_mask[0] * scale_mask / 2,
                #              size_mask[0] * scale_mask / 2)
                # offset = (offset[0] + size_mask[0] * scale_mask / 2,
                #          offset[1] + size_mask[0] * scale_mask / 2)
            
            elif action == 6:
                # Fatter
                self.size = (self.size[0] - self.alpha_w,
                             np.min((self.original_shape[1] - 1 - self.offset[1], self.size[1] + self.alpha_h)))
            elif action == 7:
                # Taller
                self.size = (np.min((self.original_shape[0] - 1 - self.offset[0], self.size[0] + self.alpha_w)),
                             self.size[1] - self.alpha_h)
            '''


            # get current image and mask
            self.region_image = update_image(self.initial_image, self.offset, self.size)
            self.region_now[int(self.offset[0]):int(self.offset[0] + self.size[0]), int(self.offset[1]):int(self.offset[1] + self.size[1])] = 1

            # calculate transformation of the boxes
            self.alpha_w = int(self.alpha * self.size[0])
            self.alpha_h = int(self.alpha * self.size[1])

            new_iou, new_p_mask = follow_iou(self.gt_mask, self.region_now)
            reward = get_reward_movement(self.iou_now, new_iou, self.p_mask_now, new_p_mask)

            self.iou_now = new_iou
            self.p_mask_now = new_p_mask
            self._update_frame_q(self.region_image)


        self.total_reward += reward
        self.previous_state = self.current_state
        #self.previous_features = self.current_features
        self.current_state = self._get_current_state()

        #print('region_size: ' + str(self.region_image.shape) + '\n')

        return reward, done, self.iou_now, self.region_now



'''
observation, reward, done, _ = self.game.step(action)

self.total_reward += reward
self._update_frame_q(observation)

self.previous_state = self.current_state
self.current_state = self._get_current_state()

return reward, done
'''