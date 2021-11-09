import sys
sys.path.insert(0, r'../')
from DQN_new_v2.deep_q_network import *

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

env_name = 'shooter'
env = make_env(env_name)
env.reset()

done = False
iter = 0
file_iter = 0
image_tensor = None
info_tensor = None
while not done:
    state, reward, done, info = env.step(-1)

    if image_tensor is None:
        image_tensor = state
    else:
        image_tensor = np.concatenate((image_tensor, state), axis=0)

    if info_tensor is None:
        info_tensor = info
    else:
        info_tensor = np.concatenate((info_tensor, info), axis=0)

    print(image_tensor.shape)
    print(info_tensor.shape)
    print('iter = ', iter)
    iter += 1
    if iter >= 10 or done:

        state_file_name = 'image_demo/image_' + str(file_iter) + '.npy'
        info_file_name = 'info_demo/info_' + str(file_iter) + '.npy'
        with open(state_file_name, 'wb') as f1:
            print(state_file_name)
            print(image_tensor.shape)
            print(info_file_name)
            print(info_tensor.shape)
            np.save(f1, image_tensor)
        with open(info_file_name, 'wb') as f2:
            np.save(f2, image_tensor)

        image_tensor = None
        info_tensor = None

        iter = 0
        file_iter += 1

    # info_stack = env.get_info_stack()
    # # print(type(info_stack))
    # print(info_stack)
    # print(info_stack.shape)
    # if len(info_stack.shape) < 2:
    #     done = True

# model = DeepQNetwork(
#     lr=0.0001,
#     n_actions=env.action_space.n,
#     input_dims=(env.observation_space.shape),
#     name='expert_network',
#     chkpt_dir='/expert'
# )


