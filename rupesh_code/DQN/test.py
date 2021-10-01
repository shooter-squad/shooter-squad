import gym
env= gym.make('PongNoFrameskip-v4')

print(env.action_space)

print(env.unwrapped.get_action_meanings())
['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
