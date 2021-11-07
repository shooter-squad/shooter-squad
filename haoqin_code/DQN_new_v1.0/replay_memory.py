import numpy as np

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape),
                                         dtype=np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)
        # * store additional states
        self.player_health_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.player_shield_availability_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.player_ultimate_availability_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.enemy_health_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.enemy_shield_availability_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.enemy_ultimate_availability_memory = np.zeros(self.mem_size, dtype=np.int64)
        # * store additional states_
        self.player_health_memory_ = np.zeros(self.mem_size, dtype=np.int64)
        self.player_shield_availability_memory_ = np.zeros(self.mem_size, dtype=np.int64)
        self.player_ultimate_availability_memory_ = np.zeros(self.mem_size, dtype=np.int64)
        self.enemy_health_memory_ = np.zeros(self.mem_size, dtype=np.int64)
        self.enemy_shield_availability_memory_ = np.zeros(self.mem_size, dtype=np.int64)
        self.enemy_ultimate_availability_memory_ = np.zeros(self.mem_size, dtype=np.int64)
    def store_transition(self, state, action, reward, state_, done, additional_state, additional_state_):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        # * store additional_state
        self.player_health_memory = additional_state[0]
        self.player_shield_availability_memory = additional_state[1]
        self.player_ultimate_availability_memory = additional_state[2]
        self.enemy_health_memory = additional_state[3]
        self.enemy_shield_availability_memory = additional_state[4]
        self.enemy_ultimate_availability_memory = additional_state[5]
        # * store additional_state_
        self.player_health_memory_ = additional_state_[0]
        self.player_shield_availability_memory_ = additional_state_[1]
        self.player_ultimate_availability_memory_ = additional_state_[2]
        self.enemy_health_memory_ = additional_state_[3]
        self.enemy_shield_availability_memory_ = additional_state_[4]
        self.enemy_ultimate_availability_memory_ = additional_state_[5]

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]
        # * store additional states
        player_health_memory = self.player_health_memory[batch]
        player_shield_availability_memory = self.player_shield_availability_memory[batch]
        player_ultimate_availability_memory = self.player_ultimate_availability_memory[batch]
        enemy_health_memory = self.enemy_health_memory[batch]
        enemy_shield_availability_memory = self.enemy_shield_availability_memory[batch]
        enemy_ultimate_availability_memory = self.enemy_ultimate_availability_memory[batch]
        # * store additional states_
        player_health_memory_ = self.player_health_memory_[batch]
        player_shield_availability_memory_ = self.player_shield_availability_memory_[batch]
        player_ultimate_availability_memory_ = self.player_ultimate_availability_memory_[batch]
        enemy_health_memory_ = self.enemy_health_memory_[batch]
        enemy_shield_availability_memory_ = self.enemy_shield_availability_memory_[batch]
        enemy_ultimate_availability_memory_ = self.enemy_ultimate_availability_memory_[batch]

        additional_state = (
            player_health_memory, 
            player_shield_availability_memory, 
            player_ultimate_availability_memory,
            enemy_health_memory,
            enemy_shield_availability_memory,
            enemy_ultimate_availability_memory
        )

        additional_state_ = (
            player_health_memory_, 
            player_shield_availability_memory_, 
            player_ultimate_availability_memory_,
            enemy_health_memory_,
            enemy_shield_availability_memory_,
            enemy_ultimate_availability_memory_
        )

        return states, actions, rewards, states_, terminal, additional_state, additional_state_
