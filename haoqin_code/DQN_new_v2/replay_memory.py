import numpy as np

import sys
sys.path.insert(0, r'../')
from Env.constants import *
import os

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions, info_stack_shape=(4, ADDITIONAL_STATE_LEN_PLAYER+ADDITIONAL_STATE_LEN_NORMAL+ADDITIONAL_STATE_LEN_CHARGE)):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape),
                                         dtype=np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)
        self.info_stack_memory = np.zeros((self.mem_size, *info_stack_shape),
                                     dtype=np.int64)
        self.new_info_stack_memory = np.zeros((self.mem_size, *info_stack_shape),
                                     dtype=np.int64)

    def store_transition(self, state, action, reward, state_, done, info_stack, info_stack_):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        info_stack_shape = info_stack.shape
        info_stack_shape_ = info_stack_.shape

        # print('INFO_STACK SHAPE: ', info_stack_shape)
        # print('INFO_STACK SHAPE_: ', info_stack_shape_)
        # self.info_stack_memory[index][0:info_stack_shape[0], 0:info_stack_shape[1]] = info_stack
        # self.new_info_stack_memory[index][0:info_stack_shape_[0], 0:info_stack_shape_[1]] = info_stack_
        self.info_stack_memory[index] = info_stack
        self.new_info_stack_memory[index] = info_stack_

        self.mem_cntr += 1

        #debug memory buffer
        print('###############DEBUG MEMORY BUFFER#################')
        print('INFO_STACK SHAPE: ', info_stack_shape)
        print('INFO_STACK SHAPE_: ', info_stack_shape_)
        print(self.state_memory[:self.mem_cntr].shape)
        print(self.new_state_memory[:self.mem_cntr].shape)
        print(self.action_memory[:self.mem_cntr].shape)
        print(self.reward_memory[:self.mem_cntr].shape)
        print(self.terminal_memory[:self.mem_cntr].shape)
        print(self.info_stack_memory[:self.mem_cntr].shape)
        print(self.new_info_stack_memory[:self.mem_cntr].shape)
        print('###############DEBUG MEMORY BUFFER#################')


    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]
        info_stack = self.info_stack_memory[batch]
        info_stack_ = self.new_info_stack_memory[batch]

        return states, actions, rewards, states_, terminal, info_stack, info_stack_

    def save_memory(self, memory_name):

        path = os.getcwd()
        new_dir = memory_name
        filename = os.path.join(path, new_dir)
        try: 
            os.mkdir(filename) 
        except OSError as error: 
            print('Warning: Memory already exist and will be overridden')  
            for f in os.listdir(memory_name):
                os.remove(os.path.join(filename, f))

        state_dir = memory_name + '/states_memory.npy'
        action_dir = memory_name + '/action_memory.npy'
        reward_dir = memory_name + '/reward_memory.npy'
        new_state_dir = memory_name + '/new_state_memory.npy'
        terminal_dir = memory_name + '/terminal_memory.npy'
        info_stack_dir = memory_name + '/info_stack_memory.npy'
        new_info_stack_dir = memory_name + '/new_info_stack_memory.npy'

        with open(state_dir, 'wb') as f:
                np.save(f, self.state_memory[:self.mem_cntr])
        with open(action_dir, 'wb') as f:
                np.save(f, self.action_memory[:self.mem_cntr])
        with open(reward_dir, 'wb') as f:
                np.save(f, self.reward_memory[:self.mem_cntr])
        with open(new_state_dir, 'wb') as f:
                np.save(f, self.new_state_memory[:self.mem_cntr])
        with open(terminal_dir, 'wb') as f:
                np.save(f, self.terminal_memory[:self.mem_cntr])
        with open(info_stack_dir, 'wb') as f:
                np.save(f, self.info_stack_memory[:self.mem_cntr])
        with open(new_info_stack_dir, 'wb') as f:
                np.save(f, self.new_info_stack_memory[:self.mem_cntr])

    def load_memory(self, N_DEMO=4):

        state_tensor = None
        action_tensor = None
        reward_tensor = None
        new_state_tensor = None
        terminal_tensor = None
        info_stack_tensor = None
        new_info_stack_tensor = None

        for i in range(N_DEMO):
            
            memory_name = 'Memory' + str(i)
            print('LOAD ' + memory_name)

            state_dir = memory_name + '/states_memory.npy'
            action_dir = memory_name + '/action_memory.npy'
            reward_dir = memory_name + '/reward_memory.npy'
            new_state_dir = memory_name + '/new_state_memory.npy'
            terminal_dir = memory_name + '/terminal_memory.npy'
            info_stack_dir = memory_name + '/info_stack_memory.npy'
            new_info_stack_dir = memory_name + '/new_info_stack_memory.npy'
    
            with open(state_dir, 'rb') as f:
                state = np.load(f)
                if state_tensor is None:
                    state_tensor = state
                else:
                    state_tensor = np.concatenate((state_tensor, state), axis=0)
            
            with open(action_dir, 'rb') as f:
                action = np.load(f)
                if action_tensor is None:
                    action_tensor = action
                else:
                    action_tensor = np.concatenate((action_tensor, action), axis=0)

            with open(reward_dir, 'rb') as f:
                reward = np.load(f)
                if reward_tensor is None:
                    reward_tensor = reward
                else:
                    reward_tensor = np.concatenate((reward_tensor, reward), axis=0)

            with open(new_state_dir, 'rb') as f:
                new_state = np.load(f)
                if new_state_tensor is None:
                    new_state_tensor = new_state
                else:
                    new_state_tensor = np.concatenate((new_state_tensor, new_state), axis=0)

            with open(terminal_dir, 'rb') as f:
                terminal = np.load(f)
                if terminal_tensor is None:
                    terminal_tensor = terminal
                else:
                    terminal_tensor = np.concatenate((terminal_tensor, terminal), axis=0)
                
            with open(info_stack_dir, 'rb') as f:
                info_stack = np.load(f)
                if info_stack_tensor is None:
                    info_stack_tensor = info_stack
                else:
                    info_stack_tensor = np.concatenate((info_stack_tensor, info_stack), axis=0)

            with open(new_info_stack_dir, 'rb') as f:
                new_info_stack = np.load(f)
                if new_info_stack_tensor is None:
                    new_info_stack_tensor = new_info_stack
                else:
                    new_info_stack_tensor = np.concatenate((new_info_stack_tensor, new_info_stack), axis=0)
    
            print('###############DEBUG LOAD MEMORY#################')
            print(state_tensor.shape)
            print(action_tensor.shape)
            print(reward_tensor.shape)
            print(new_state_tensor.shape)
            print(terminal_tensor.shape)
            print(info_stack_tensor.shape)
            print(new_info_stack_tensor.shape)
            print('###############DEBUG LOAD MEMORY#################')

        index = state_tensor.shape[0]
        self.state_memory[:index] = state_tensor
        self.action_memory[:index] = action_tensor
        self.reward_memory[:index] = reward_tensor
        self.new_state_memory[:index] = new_state_tensor
        self.terminal_memory[:index] = terminal_tensor
        self.info_stack_memory[:index] = info_stack_tensor
        self.new_info_stack_memory[:index] = new_info_stack_tensor
        
        self.mem_cntr = index
        print('SELF.MEM_CNTR IS: ', self.mem_cntr)
        np.set_printoptions(threshold=sys.maxsize)
        print(self.new_info_stack_memory)
        # self.state_memory[index] = state
        # self.new_state_memory[index] = state_
        # self.action_memory[index] = action
        # self.reward_memory[index] = reward
        # self.terminal_memory[index] = done

        # info_stack_shape = info_stack.shape
        # info_stack_shape_ = info_stack_.shape

        # print('INFO_STACK SHAPE: ', info_stack_shape)
        # print('INFO_STACK SHAPE_: ', info_stack_shape_)


        # self.info_stack_memory[index][0:info_stack_shape[0], 0:info_stack_shape[1]] = info_stack
        # self.new_info_stack_memory[index][0:info_stack_shape_[0], 0:info_stack_shape_[1]] = info_stack_


        # self.mem_cntr += 1