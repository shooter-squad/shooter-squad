import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(DeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        # self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4)
        self.conv1 = nn.Conv2d(1, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        # fc_input_dims = self.calculate_conv_output_dims(input_dims)
        fc_input_dims = 3136
        # self.fc1 = nn.Linear(fc_input_dims, 512)

        self.rnn= nn.LSTM(fc_input_dims, hidden_size=512)

        self.fc2 = nn.Linear(512, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)

        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        print(self.device)
        self.to(self.device)

    # def calculate_conv_output_dims(self, input_dims):
    #     state = T.zeros(1, *input_dims)
    #     dims = self.conv1(state)
    #     dims = self.conv2(dims)
    #     dims = self.conv3(dims)
    #     return int(np.prod(dims.size()))

    def forward(self, state):
        rnn_input_list = []
        for t in range(4):
            # * state.shape:  torch.Size([64, 4, 84, 84])
            # * state_t.shape:  torch.Size([64, 1, 84, 84])
            state_t = state[:, t, :, :].unsqueeze(1)
            conv1 = F.relu(self.conv1(state_t))
            conv2 = F.relu(self.conv2(conv1))
            conv3 = F.relu(self.conv3(conv2))
            # conv3 shape is BS x n_filters x H x W
            conv_state = conv3.view(conv3.size()[0], -1)
            # conv_state shape is BS x (n_filters * H * W)
            # * conv_state.shape is: torch.Size([64, 3136])
            rnn_input_list.append(conv_state)
        # * conv_state.shape is: torch.Size([4, 64, 3136])
        rnn_input = T.stack(rnn_input_list, 0)
        # * flat1.shape is: torch.Size([1, 64, 512])
        # print('rnn_input shape is ', rnn_input.shape)
        x, (flat1, c) = self.rnn(rnn_input)
        # * flat1.shape is: torch.Size([64, 512])
        flat1 = T.squeeze(flat1, 0)
        # flat1 = F.relu(self.fc1(conv_state))
        

        actions = self.fc2(flat1) # // NOTE: actions: [32, 6]

        return actions

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))
