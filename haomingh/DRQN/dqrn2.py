import random
import math
import gym
import numpy as np
import PIL
from PIL import Image
import matplotlib
import matplotlib.cm as cm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import sys
# adding Folder_2 to the system path
sys.path.insert(0, r'/home/haoqindegcp/shooter-squad/haoqin_code')
from Env import *

class model(nn.Module):
    def __init__(self):
        super(model,self).__init__()
        self.hidden_size = 512
        self.conv1=nn.Conv2d(4,32,kernel_size=8,stride=4)
        self.bn1=nn.BatchNorm2d(32)
        self.conv2=nn.Conv2d(32,64,kernel_size=4,stride =2)
        self.bn2=nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.rnn= nn.RNN(input_size=64*7*7, hidden_size=512,num_layers=2,batch_first=True)
        self.fc = nn.Linear(512, 2)
        
    def init_hidden(self,batch_size):
        return (torch.zeros(2,batch_size, self.hidden_size))
    
    def forward(self,x,hidden):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x=x.reshape(x.shape[0],1,7*7*64)
        print(x.shape)
        x,h_0=self.rnn(x,hidden)
        return self.fc(x.contiguous().view(x.size(0), -1))

env = ShooterEnv()
env.reset()

policy=model()
target_net=model()
target_net.load_state_dict(policy.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy.parameters())
criterion = F.smooth_l1_loss

memory=10000
store=[[dict()] for i in range(memory)]
gamma=0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

def PIL2array(img):
    return np.array(img.getdata(),np.uint8).reshape(img.size[1], img.size[0], 4)

def array2PIL(arr, size):
    mode = 'RGBA'
    arr = arr.reshape(arr.shape[0]*arr.shape[1], arr.shape[2])
    if len(arr[0]) == 3:
        arr = np.c_[arr, 255*np.ones((len(arr),1), np.uint8)]
    return Image.frombuffer(mode, size, arr.tostring(), 'raw', mode, 0, 1)

def processScreen(screen):
    s=[600,400]
    image= array2PIL(screen,s)
    newImage = image.resize((84, 84))
    xtt=PIL2array(newImage)
    xtt=xtt.reshape(xtt.shape[2],xtt.shape[0],xtt.shape[1])
    img=torch.from_numpy(np.array(xtt))
    img=img.type('torch.FloatTensor')
    return img/255.0

def addEpisode(ind,prev,curr,reward,act):
    if len(store[ind]) ==0:
        store[ind][0]={'prev':prev,'curr':curr,'reward':reward,'action':act}
    else:
        store[ind].append({'prev':prev,'curr':curr,'reward':reward,'action':act})

def trainNet(total_episodes):
    if total_episodes==0:
        return
    ep=random.randint(0,total_episodes-1)
    if len(store[ep]) < 8:
        return
    else:  
        start=random.randint(1,len(store[ep])-1)
        length=len(store[ep])
        inp=[]
        target=[]
        rew=torch.Tensor(1,length-start)
        actions=torch.Tensor(1,length-start)
        
        for i in range(start,length,1):
            inp.append((store[ep][i]).get('prev'))
            target.append((store[ep][i]).get('curr'))
            rew[0][i-start]=store[ep][i].get('reward')
            actions[0][i-start]=store[ep][i].get('action')
        targets = torch.Tensor(target[0].shape[0],target[0].shape[1],target[0].shape[2])
        torch.cat(target, out=targets)
        ccs=torch.Tensor(inp[0].shape[0],inp[0].shape[1],inp[0].shape[2])
        torch.cat(inp, out=ccs)
        hidden = policy.init_hidden(length-start)
        qvals= target_net(targets,hidden)
        actions=actions.type('torch.LongTensor')
        actions=actions.reshape(length-start,1)
        hidden = policy.init_hidden(length-start)
        inps=policy(ccs,hidden).gather(1,actions)
        p1,p2=qvals.detach().max(1)
        targ = torch.Tensor(1,p1.shape[0])   
        for num in range(start,length,1):
            if num==len(store[ep])-1:
                targ[0][num-start]=rew[0][num-start] 
            else:
                targ[0][num-start]=rew[0][num-start]+gamma*p1[num-start]
        optimizer.zero_grad()
        inps=inps.reshape(1,length-start)
        loss = criterion(inps,targ)
        loss.backward()
        for param in policy.parameters():
            param.grad.data.clamp(-1,1)
        optimizer.step()

def trainDRQN(episodes):
    steps_done=0
    for i in range(0,episodes,1):        
        print("Episode",i)
        env.reset()
        prev=env.render(mode='rgb_array')
        prev=processScreen(prev)
        done=False
        steps=0
        rew=0
        while done == False:
            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY)
            print(steps,end=" ")
            steps+=1
            hidden = policy.init_hidden(1)
            output=policy(prev.unsqueeze(0),hidden)
            action=(output.argmax()).item()
            rand= random.uniform(0,1)
            if rand < 0.05:
                action=random.randint(0,1)

            _,reward,done,_=env.step(action)   
            rew=rew+reward
            if steps>200:
                terminal = torch.zeros(prev.shape[0],prev.shape[1],prev.shape[2])
                addEpisode(i,prev.unsqueeze(0),terminal.unsqueeze(0),-10,action)
                f=0
                break
            sc=env.render(mode='rgb_array')
            sc=processScreen(sc)
            addEpisode(i,prev.unsqueeze(0),sc.unsqueeze(0),reward,action)
            trainNet(i)
            prev=sc
            steps_done+=1
        terminal = torch.zeros(prev.shape[0],prev.shape[1],prev.shape[2])
        print(rew)
        addEpisode(i,prev.unsqueeze(0),terminal.unsqueeze(0),-10,action)
        if i%10==0:
            target_net.load_state_dict(policy.state_dict())

trainDRQN(2000)