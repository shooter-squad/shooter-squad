import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import sys
sys.path.insert(0, r'../')
from Env.constants import *

class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir, pre_train_dir, info_stack_dims=(4, ADDITIONAL_STATE_LEN_PLAYER+ADDITIONAL_STATE_LEN_NORMAL+ADDITIONAL_STATE_LEN_CHARGE)):
        super(DeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        self.pre_train_file = os.path.join(pre_train_dir, name)
        
        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        fc_input_dims = self.calculate_conv_output_dims(input_dims) + info_stack_dims[0] * info_stack_dims[1]
        # print(fc_input_dims)

        self.fc1 = nn.Linear(fc_input_dims, 512)
        self.fc2 = nn.Linear(512, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)

        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        print(self.device)
        self.to(self.device)

    def calculate_conv_output_dims(self, input_dims):
        state = T.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))

    def forward(self, state, info_stack):
        # * N=batch_size=64, C_in=input_channel=4, H=84, W=84
        print('ENTERING FORWARD')
        print('state.shape: ', state.shape)
        conv1 = F.relu(self.conv1(state))
        print('conv1.shape: ', conv1.shape)
        conv2 = F.relu(self.conv2(conv1))
        print('conv2.shape: ', conv2.shape)
        conv3 = F.relu(self.conv3(conv2))
        print('conv3.shape: ', conv3.shape)
        # conv3 shape is BS x n_filters x H x W
        conv_state = conv3.view(conv3.size()[0], -1)
        print('conv_state.shape: ', conv_state.shape)
        # * flatten info_stack
        info_stack_flatten = info_stack.view(info_stack.size()[0], -1)
        print('info_stack_flatten.shape: ', conv_state.shape)
        flatten_cat = T.cat([conv_state, info_stack_flatten], dim=1)
        
        print('flatten_cat.shape: ', flatten_cat.shape)
        # conv_state shape is BS x (n_filters * H * W)
        flat1 = F.relu(self.fc1(flatten_cat))
        print('flat1.shape: ', flat1.shape)
        actions = self.fc2(flat1) # // NOTE: actions: [32, 6]
        print('actions.shape: ', actions.shape)

        return actions

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

    def pre_train(self):
        print('... loading pre_train model from ' + str(self.pre_train_file) + ' ...')

        pre_train_state = T.load(self.pre_train_file, map_location='cpu')
        print(type(pre_train_state))
        cur_state_dict = self.state_dict()  # The current model's state_dict

        for key in pre_train_state.keys():
            if key in cur_state_dict:
                if pre_train_state[key].shape == cur_state_dict[key].shape:
                    cur_state_dict[key] = nn.Parameter(pre_train_state[key].data)
                    print('Success: Loaded {} from pre-train model'.format(key))
                else:
                    print('Error: Size mismatch for {}, size of loaded model {}, size of current model {}'.format(
                        key, pre_train_state[key].shape, cur_state_dict[key].shape))
                    temp = cur_state_dict[key].numpy()
                    pre_arr = pre_train_state[key].numpy()
                    if len(temp.shape) == 2:
                        temp[0:pre_arr.shape[0], 0:pre_arr.shape[1]] = pre_arr
                    elif len(temp.shape) == 1:
                        temp[0:pre_arr.shape[0]] = pre_arr
                    cur_state_dict[key] = nn.Parameter(T.from_numpy(temp))

            else:
                print('Error: Loaded weight {} not present in current model'.format(key))

        # self.load_state_dict(cur_state_dict)
