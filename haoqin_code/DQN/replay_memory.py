import numpy as np

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions, info_stack_shape=(4, 6)):
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
        self.info_stack_memory[index] = info_stack
        self.new_info_stack_memory[index] = info_stack_


        self.mem_cntr += 1

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
