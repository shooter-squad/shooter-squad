import sys

import numpy as np

from actor_critic import ActorCriticAgent
from utils import plot_learning_curve, make_env

# Adds Env folder to the system path
sys.path.insert(0, r'/home/zhuli/projects/shooter-squad/lihan_code')

if __name__ == '__main__':
    env_name = 'shooter'
    env = make_env(env_name)

    agent = ActorCriticAgent(gamma=0.99, lr=5e-6, input_dims=(env.observation_space.shape),
                             n_actions=env.action_space.n, fc1_dims=2048, fc2_dims=1536)
    n_games = 4000

    fname = 'ACTOR_CRITIC_' + str(agent.fc1_dims) + \
            '_fc1_dims_' + str(agent.fc2_dims) + '_fc2_dims_lr' + str(agent.lr) + \
            '_' + str(n_games) + 'games'
    figure_file = 'plots/' + fname + '.png'

    scores = []
    for i in range(n_games):
        done = False
        observation = env.reset()
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.learn(observation, reward, observation_, done)
            observation = observation_
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print('episode ', i, 'score %.1f' % score,
              'average score %.1f' % avg_score)

    x = [i + 1 for i in range(n_games)]
    plot_learning_curve(x, scores, figure_file)
