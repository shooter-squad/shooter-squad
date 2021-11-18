import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import sys
# sys.path.insert(0, r'../')
from Env.constants import *

class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir, info_stack_dims=(4, ADDITIONAL_STATE_LEN_PLAYER+ADDITIONAL_STATE_LEN_NORMAL+ADDITIONAL_STATE_LEN_CHARGE)):
        super(DeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        
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

    def PreTrain(self):
        print('... loading pre_train model ...')
        self.load_state_dict(T.load(self.checkpoint_file))

class DeepQNetwork2(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir, info_stack_dims=(4, ADDITIONAL_STATE_LEN_PLAYER+ADDITIONAL_STATE_LEN_NORMAL+ADDITIONAL_STATE_LEN_CHARGE)):
        super(DeepQNetwork2, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        
        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        # * this is for extracting features of info stack
        self.st_fc1 = nn.Linear(info_stack_dims[0] * info_stack_dims[1], 64)
        self.st_fc2 = nn.Linear(64, 32)

        fc_input_dims = self.calculate_conv_output_dims(input_dims) + 32
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

        # * this is for extracting features of info stack
        # # * flatten info_stack
        info_stack_flatten = info_stack.view(info_stack.size()[0], -1).float()
        st_fc1 = F.relu(self.st_fc1(info_stack_flatten))
        st_fc2 = F.relu(self.st_fc2(st_fc1))
        
        # info_stack_flatten = info_stack.view(info_stack.size()[0], -1)
        # print('info_stack_flatten.shape: ', conv_state.shape)
        flatten_cat = T.cat([conv_state, st_fc2], dim=1)
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

    def PreTrain(self):
        print('... loading pre_train model ...')
        self.load_state_dict(T.load(self.checkpoint_file))

if __name__ == "__main__":
    model = DeepQNetwork2(lr=0.1, n_actions=4, name='test', input_dims=(4, 84, 84), chkpt_dir='test_checkpoint')

    import torch
    from torchviz import make_dot

    x = torch.randn(1, 4, 84, 84).cuda()
    info_stack = torch.randn(1, 4, ADDITIONAL_STATE_LEN_PLAYER+ADDITIONAL_STATE_LEN_NORMAL+ADDITIONAL_STATE_LEN_CHARGE).cuda()
    y = model(x, info_stack)

    make_dot(y.mean(), params=dict(model.named_parameters())).render("dqn_network_graph_cat_v2.0", format="png")