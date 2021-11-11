import sys
sys.path.insert(0, r'../')
from DQN_new_v2.deep_q_network import *
 
import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, r'../')
from Env.constants import *
from Env.utils import make_env

from torchvision.utils import save_image
import torch
import torchvision
 
device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

'''
MODE:{'DEMO', 'PRETRAIN'}
'''
MODE = 'DEMO'
SAVE_IMG = False
BATCH_SIZE = 64
N_EPOCH = 5
 
GAME = 0
 
N_DEMO = 15
 
PARTIAL_BATCH_SIZE = 8
 

 
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
    action_tensor = None
    action_list = []
 
    n_steps = 0
 
    while not done:
        state, reward, done, info = env.step(-1)
        info = env.get_info_stack()
        state = np.expand_dims(state, axis=0)
        info = np.expand_dims(info, axis=0)
 
        action = env.player_action_num
       
        if image_tensor is None:
            image_tensor = state
        else:
            image_tensor = np.concatenate((image_tensor, state), axis=0)
 
        if info_tensor is None:
            info_tensor = info
        else:
            info_tensor = np.concatenate((info_tensor, info), axis=0)
 
        action_list.append(action)
 
        # print(image_tensor.shape)
        # print(info_tensor.shape)
        # print('iter = ', iter)
        iter += 1
        if iter >= PARTIAL_BATCH_SIZE or done:
 
            action_tensor = T.Tensor(action_list)
 
            state_file_name = game_name + '/image_demo/image_' + str(file_iter) + '.npy'
            info_file_name = game_name + '/info_demo/info_' + str(file_iter) + '.npy'
            action_file_name = game_name + '/action_demo/info_' + str(file_iter) + '.npy'
                # print(state_file_name)
                # print(image_tensor.shape)
                # print(info_file_name)
                # print(info_tensor.shape)
            with open(state_file_name, 'wb') as f1:
                np.save(f1, image_tensor)
            with open(info_file_name, 'wb') as f2:
                np.save(f2, info_tensor)
            with open(action_file_name, 'wb') as f2:
                np.save(f2, action_tensor)
 
            image_tensor = None
            info_tensor = None
            action_list = []
            action_tensor = None
 
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
                # print(info_dir + filename)
 
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
        print(action_tensor.shape)
 
    if SAVE_IMG:
        for i in range(image_tensor.shape[0]):
            image = T.Tensor(image_tensor[i,0,:,:].squeeze())
            image_name = 'DemoImages/img' + str(i) + '.png'
            save_image(image, image_name)

    '''
    image_tensor: (818, 4, 84, 84)
    info_tensor: (818, 4, 27)
    action_tensor: (818,)
    '''
    image_tensor = T.Tensor(image_tensor).to(device)
    info_tensor = T.Tensor(info_tensor).to(device)
    action_tensor = T.Tensor(action_tensor).long().to(device)

    # image_tensor = image_tensor.contiguous().view(-1, 4, 84, 84)
    # print(info_tensor.shape)

    network = DeepQNetwork(
        lr=0.001, 
        n_actions=env.action_space.n,
        input_dims=env.observation_space.shape,
        name = env_name + 'v2',
        chkpt_dir='pretrain/'
    )

    optimizer = optim.Adam(network.parameters(), lr = 0.01)

    def get_num_correct(preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()

    

    total_len = image_tensor.shape[0] // BATCH_SIZE

    for j in range(N_EPOCH):
        print('Epoch ' + str(j) + ' starts......')
        total_loss = 0
        total_correct = 0
        for iter in range(total_len):
            images = image_tensor[iter * BATCH_SIZE:(iter + 1) * BATCH_SIZE,:,:,:]
            infos = info_tensor[iter * BATCH_SIZE:(iter + 1) * BATCH_SIZE,:,:]
            labels = action_tensor[iter * BATCH_SIZE:(iter + 1) * BATCH_SIZE]
            # images, labels = batch
            preds = network(images, infos)
            loss = F.cross_entropy(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_correct += get_num_correct(preds, labels)
            # print('Total correct is: ', total_correct)

        print('Accuracies: ', total_correct / (total_len * BATCH_SIZE))