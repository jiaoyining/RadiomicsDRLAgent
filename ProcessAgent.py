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

from datetime import datetime
from multiprocessing import Process, Queue, Value

import numpy as np
import time

from Config import Config
#from BrainEnvironment import Environment
from Experience import Experience
import gym
from ToolsFunc import *



class ProcessAgent(Process):
    def __init__(self, id, prediction_q, training_q, episode_log_q):
        super(ProcessAgent, self).__init__()

        self.id = id
        self.prediction_q = prediction_q
        self.training_q = training_q
        self.episode_log_q = episode_log_q

        self.env = gym.make('BrainEnvironment-v0', Config=Config)

        self.env = self.env.unwrapped
        self.num_actions = self.env.get_num_actions()
        self.actions = np.arange(self.num_actions)

        self.discount_factor = Config.DISCOUNT
        # one frame at a time
        self.wait_q = Queue(maxsize=10)
        self.exit_flag = Value('i', 0)
    '''
    def start(self):
        self.run()
    '''

    @staticmethod
    def _accumulate_rewards(experiences, discount_factor, terminal_reward):
        reward_sum = terminal_reward
        for t in reversed(range(0, len(experiences))):
            r = experiences[t].reward
            reward_sum = discount_factor * reward_sum + r
            experiences[t].reward = reward_sum
        return experiences[:-1]

    def convert_data(self, experiences):
        x_ = np.array([exp.state for exp in experiences])
        a_ = np.eye(self.num_actions)[np.array([exp.action for exp in experiences]).astype('int')].astype(np.float32)
        #print(str(a_) + '\n')
        r_ = np.array([exp.reward for exp in experiences])
        r0_ = np.array([exp.reward0 for exp in experiences])
        p_mask_ = np.array([exp.iou for exp in experiences])
        return x_, r_, r0_, a_, p_mask_

    def predict(self, state):
        # put the state in the prediction q
        self.prediction_q.put((self.id, state))
        # wait for the prediction to come back
        p, v = self.wait_q.get()
        return p, v

    def select_action(self, prediction):
        if Config.PLAY_MODE:
            action = np.argmax(prediction)
        else:
            action = np.random.choice(self.actions, p=prediction).astype('int')
        return action

    def run_episode(self):
        #print('reseting \n')
        self.env.reset()
        done = False
        experiences = []
        time_count = 0

        while (not done) and time_count <= Config.TIME_MAX:

            # very first few frames
            if len(self.env.current_state) == 0:
                reward, done, p_mask, region_now = self.env.step(np.random.randint(0, 4, 1))
                #print('reset \n')
                #self.env.step(5)  #0 == NOOP
                if Config.PLAY_MODE and done == False:
                    draw_gif_sequences_test(time_count, region_now, self.env.img_name, save_boolean=1)
                elif Config.PLAY_MODE and done == True:
                    drawing_gif(self.env.img_name)

                time_count += 1

                continue

            #print('shape: ' + str(self.env.current_state) + '\n')
            prediction, value = self.predict(self.env.current_state)
            action = self.select_action(prediction)

            if time_count < Config.TIME_MAX:
                #action = 5
                reward, done, p_mask, region_now = self.env.step(action)
                exp = Experience(self.env.previous_state, action, reward, reward, done, p_mask)
                experiences.append(exp)

            else:
                reward, done, p_mask, region_now = self.env.step(4)
                exp = Experience(self.env.previous_state, 4, reward, reward, done, p_mask)
                experiences.append(exp)


            if Config.PLAY_MODE and done == False:
                draw_gif_sequences_test(time_count, region_now, self.env.img_name, save_boolean=1)
            elif Config.PLAY_MODE and done == True:
                drawing_gif(self.env.img_name)


            if done:
                terminal_reward = 0 if done else value
                updated_exps = ProcessAgent._accumulate_rewards(experiences, self.discount_factor, terminal_reward)
                x_, r_, r0_, a_, p_mask_ = self.convert_data(updated_exps)
                #print('time: ' + str(time_count) + ', done: ' + str(done) + ', p_mask: ' + str(p_mask) + ', reward: ' + str(reward_all) + ', action:' + str(action) + ' ,a_:' + str(a_.shape) + '\n')
                # keep the last experience for the next batch
                experiences = []
                yield x_, r_, r0_, a_, p_mask_, self.env.img_name

            time_count += 1

    def run(self):

        # randomly sleep up to 1 second. helps agents boot smoothly.
        time.sleep(np.random.rand())
        np.random.seed(np.int32(time.time() % 1 * 1000 + self.id * 10))

        while self.exit_flag.value == 0:
            total_reward = 0
            total_length = 0

            for x_, r_, r0_, a_, p_mask_, name in self.run_episode():

                total_length += len(r_) + 1  # +1 for last frame that we drop

                self.training_q.put((x_, r_, a_))
                self.episode_log_q.put((datetime.now(), np.sum(r_), total_length, p_mask_[-1], str(r_[-1]), str(r_), str(p_mask_)))

                #if p_mask_[-1] >= 0.7 and np.sum(r0_) < 0:
                #    print(str(r0_) + '\n')