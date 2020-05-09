import numpy as np
'''
from keras.models import Sequential
from keras import initializers
from keras.initializers import normal, identity
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop, SGD, Adam
'''
from tensorflow import keras as krs
import tensorflow as tf
import SimpleITK as sitk


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
reward_terminal_action = 3
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


def get_reward_movement(iou, new_iou, p_mask, new_p_mask):

    if new_iou > iou:
        reward = reward_movement_action
    else:
        reward = -reward_movement_action


    #reward = (new_iou - iou) * 10

    return reward


def get_reward_trigger(new_iou, new_p_mask, initial_iou):
    '''
    if new_iou > 0.8:
        reward = (new_iou - initial_iou) * 10
    else:
        reward = (new_iou - initial_iou) * 10
    '''
    if new_iou > 0.8:
        reward = reward_terminal_action
    else:
        reward = -reward_terminal_action



    return reward

