import sys
sys.path.insert(0, r'../')
from DQN_v2_pre_train.deep_q_network import *

import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import sys
sys.path.insert(0, r'../')
from Env.constants import *
from Env.utils import make_env

device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

GAME = 0

N_DEMO = 1

PARTIAL_BATCH_SIZE = 8

MODE = 'DEMO'

game_name = 'Demo' + str(GAME)

env_name = 'shooter'
env = make_env(env_name)
env.reset()



if MODE == 'DEMO':

    done = False
    iter = 0
    file_iter = 0
    image_tensor = None
    info_tensor = None

    n_steps = 0

    while not done:
        state, reward, done, info = env.step(-1)
        info = env.get_info_stack()
        state = np.expand_dims(state, axis=0)
        info = np.expand_dims(info, axis=0)

        action = env.get_player_action()
        
        if image_tensor is None:
            image_tensor = state
        else:
            image_tensor = np.concatenate((image_tensor, state), axis=0)

        if info_tensor is None:
            info_tensor = info
        else:
            info_tensor = np.concatenate((info_tensor, info), axis=0)

        # print(image_tensor.shape)
        # print(info_tensor.shape)
        # print('iter = ', iter)
        iter += 1
        if iter >= PARTIAL_BATCH_SIZE or done:

            state_file_name = game_name + '/image_demo/image_' + str(file_iter) + '.npy'
            info_file_name = game_name + '/info_demo/info_' + str(file_iter) + '.npy'
            with open(state_file_name, 'wb') as f1:
                # print(state_file_name)
                # print(image_tensor.shape)
                # print(info_file_name)
                # print(info_tensor.shape)
                np.save(f1, image_tensor)
            with open(info_file_name, 'wb') as f2:
                np.save(f2, info_tensor)

            image_tensor = None
            info_tensor = None

            iter = 0
            file_iter += 1

        n_steps += 1

    print('n_steps: ', n_steps)

elif MODE == 'PRETRAIN':
    image_tensor = None
    info_tensor = None
    action_tensor = None

    model = DeepQNetwork(
        lr=0.001, 
        n_actions=env.action_space.n,
        input_dims=env.observation_space.shape,
        name = env_name + 'v2',
        chkpt_dir='pretrain/'
    )

    for i in range(N_DEMO):
        
        image_dir = 'Demo' + str(i) + '/image_demo/'
        info_dir = 'Demo' + str(i) + '/info_demo/'
        action_dir = 'Demo' + str(i) + '/action_demo/'

        # print(image_dir)

        for filename in os.listdir(image_dir):
            # print(filename)
            with open(image_dir + filename, 'rb') as f1:
                image = np.load(f1)

                if image_tensor is None:
                    image_tensor = image
                else:
                    image_tensor = np.concatenate((image_tensor, image), axis=0)

        for filename in os.listdir(info_dir):
            with open(info_dir + filename, 'rb') as f2:
                print(info_dir + filename)

                info = np.load(f2)

                if info_tensor is None:
                    info_tensor = info
                else:
                    info_tensor = np.concatenate((info_tensor, info), axis=0)

        for filename in os.listdir(action_dir):
            with open(action_dir + filename, 'rb') as f2:
                action = np.load(f2)

                if action_tensor is None:
                    action_tensor = action
                else:
                    action_tensor = np.concatenate((action_tensor, action), axis=0)



            # with open(action_dir + filename, 'rb') as f3:
            #     action = np.load(f3)
        print(image_tensor.shape)
        print(info_tensor.shape)
        # print(action_tensor.shape)


# image_tensor = T.Tensor(image_tensor).to(device)
# # info_tensor = T.Tensor(info_tensor).to(device)
# # action_tensor = T.Tensor(action_tensor).to(device)

# image_tensor = image_tensor.contiguous().view(-1, 4, 8, 8)
# print(info_tensor.shape)