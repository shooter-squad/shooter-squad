# import gym
import numpy as np
from dqn_agent import DQNAgent
from utils import plot_learning_curve, make_env
# from gym import wrappers
import time
import sys
# adding Folder_2 to the system path
sys.path.insert(0, r'/home/elvinzhu/github/shooter-squad/haoyun_code')
from newEnv import *

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