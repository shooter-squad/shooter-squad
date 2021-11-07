# import gym
import numpy as np
from dqn_agent import DQNAgent
from utils import plot_learning_curve, make_env
# from gym import wrappers
import time
import sys
# adding Folder_2 to the system path
sys.path.insert(0, r'C:\Users\haoqi\OneDrive\Desktop\shooter-squad\haoqin_code')
from Env import *

if __name__ == '__main__':
    env = ShooterEnv()

    action_list = [
        0, 1, 2, 3, 0, 0, 1, 2, 3, 0, 0, 1, 2, 3, 0, 0, 1, 2, 3, 0, 0, 1, 2, 3, 0
    ]

    for action in action_list:
        env.step(action)
        env.step(action)
        env.step(action)
        env.step(action)
        env.step(action)