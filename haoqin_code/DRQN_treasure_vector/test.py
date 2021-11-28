import gym
import torch
from torch import nn


# env= gym.make('PongNoFrameskip-v4')

# print(env.action_space)

# print(env.unwrapped.get_action_meanings())
# ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']

A = torch.Tensor(2,2,2,2)
print(A.shape)
B = A[:, 1, :, :]
print(B.shape)
C = A[:, 1, :, :].unsqueeze(1)
print(C.shape)
C = A[:, 0, :, :]
D = []
D.append(C)
D.append(B)
D.append(C)
print(D)

E = torch.stack(D, 0)
print(E.shape)

rnn = nn.LSTM(10, 20) # * input_size, hidden_size, num_layer
input = torch.randn(5, 3, 10) # * lengh_seq, batch_size, input_size
h0 = torch.randn(2, 3, 20)
c0 = torch.randn(2, 3, 20)
output, (hn, cn) = rnn(input)
print(output.shape)
print(hn.shape)

hn = torch.squeeze(hn, 0)
print(hn.shape)

