import numpy as np
'''
from keras.models import Sequential
from keras import initializers
from keras.initializers import normal, identity
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop, SGD, Adam
'''
from features import *
from tensorflow import keras as krs
import tensorflow as tf
import SimpleITK as sitk

tf.enable_eager_execution()

# Different actions that the agent can do
number_of_actions = 9
# Actions captures in the history vector
actions_of_history = 15
# Visual descriptor size
visual_conv_descriptor_size = 25088
# Reward movement action
reward_movement_action = 1
# p_mask
reward_pmask_action = 1
# Reward terminal action
reward_terminal_action = 15
# t_p_mask
reward_t_pmask_action = 1
# IoU required to consider a positive detection
iou_threshold = 0.7


def update_history_vector(history_vector, action):
    action_vector = np.zeros(number_of_actions)
    action_vector[action-1] = 1
    size_history_vector = np.size(np.nonzero(history_vector))
    updated_history_vector = np.zeros(number_of_actions*actions_of_history)
    if size_history_vector < actions_of_history:
        aux2 = 0
        for l in range(number_of_actions*size_history_vector, number_of_actions*size_history_vector+number_of_actions - 1):
            history_vector[l] = action_vector[aux2]
            aux2 += 1
        return history_vector
    else:
        for j in range(0, number_of_actions*(actions_of_history-1) - 1):
            updated_history_vector[j] = history_vector[j+number_of_actions]
        aux = 0
        for k in range(number_of_actions*(actions_of_history-1), number_of_actions*actions_of_history):
            updated_history_vector[k] = action_vector[aux]
            aux += 1
        return updated_history_vector


def get_state(image, history_vector, model):
    '''
    conv_maps = get_conv_image_descriptor_for_image(image, model)
    descriptor_image = np.zeros((visual_conv_descriptor_size*4, 1))

    for i_map in range(4):
        descriptor_image[i_map*visual_conv_descriptor_size:(i_map+1)*visual_conv_descriptor_size, :] \
            = np.reshape(conv_maps[i_map], (visual_conv_descriptor_size, 1))
    '''

    radiomics_descriptor = get_radiomics_descriptor_for_image(image)
    history_vector = np.reshape(history_vector, (number_of_actions*actions_of_history, 1))
    state = np.vstack((radiomics_descriptor, history_vector))
    return state


def get_state_pool45(history_vector,  region_descriptor):
    history_vector = np.reshape(history_vector, (24, 1))
    return np.vstack((region_descriptor, history_vector))


def get_reward_movement(iou, new_iou, p_mask, new_p_mask):

    if new_p_mask > p_mask:
        reward = reward_movement_action
    else:
        reward = - reward_movement_action
    return reward


def get_reward_trigger(new_iou, new_p_mask):
    if new_p_mask > iou_threshold:
        reward = reward_terminal_action
    else:
        reward = - reward_terminal_action
    return reward


def get_q_network(weights_path):
    model = krs.models.Sequential()
    #model.add(krs.layers.BatchNormalization(axis=-1))
    model.add(krs.layers.Dense(1024, kernel_initializer="uniform", input_shape=(91*4+15*9, )))
    model.add(krs.layers.Activation('relu'))
    #model.add(krs.layers.BatchNormalization(axis=-1))
    model.add(krs.layers.Dropout(0.2))

    model.add(krs.layers.Dense(1024, kernel_initializer="uniform"))
    model.add(krs.layers.Activation('relu'))
    #model.add(krs.layers.BatchNormalization(axis=-1))
    model.add(krs.layers.Dropout(0.2))

    model.add(krs.layers.Dense(9, kernel_initializer="uniform"))
    model.add(krs.layers.Activation('linear'))
    adam = tf.train.AdamOptimizer(1e-6)
    #model = multi_gpu_model(model, gpus=2)
    model.compile(loss='mse', optimizer=adam)
    if weights_path != "0":
        model.load_weights(weights_path)
    return model



def get_array_of_q_networks_for_pascal(weights_path, class_object):
    q_networks = []
    if weights_path == "0":
        for i in range(5):
            q_networks.append(get_q_network("0"))
    else:
        for i in range(5):
            if i == (class_object-1):
                q_networks.append(get_q_network(weights_path + "model" + str(i+1) + "h5.index"))
            else:
                q_networks.append(get_q_network("0"))
    return np.array([q_networks])[0][class_object-1]
