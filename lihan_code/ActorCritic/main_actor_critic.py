import sys

# Adds Env folder to the system path
sys.path.insert(0, r'C:\Users\Nathan\Documents\GitHub\shooter-squad\lihan_code')

import numpy as np

from actor_critic import ActorCriticAgent
from utils import plot_learning_curve, make_env

if __name__ == '__main__':
    env_name = 'shooter'
    env = make_env(env_name)

    agent = ActorCriticAgent(gamma=0.99, lr=5e-6, input_dims=(env.observation_space.shape),
                             n_actions=env.action_space.n, fc1_dims=512, fc2_dims=256, chkpt_dir='models/')

    n_games = 3000
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()

    output_file = open('stats.txt', 'w')

    fname = 'ACTOR_CRITIC_' + str(agent.fc1_dims) + \
            '_fc1_dims_' + str(agent.fc2_dims) + '_fc2_dims_lr' + str(agent.lr) + \
            '_' + str(n_games) + 'games'
    figure_file = 'plots/' + fname + '.png'

    n_steps = 0
    best_score = -np.inf
    scores = []

    for i in range(n_games):
        done = False
        observation = env.reset()
        score = 0
        cur_step = 0

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.learn(observation, reward, observation_, done)
            observation = observation_

            cur_step += 1

            if cur_step >= 3000:
                done = True

        if cur_step >= 3000:
            print('Error: game steps too big.')
            continue

        n_steps += cur_step

        scores.append(score)

        avg_score = np.mean(scores[-100:])

        if avg_score > best_score:
            if not load_checkpoint:
                agent.save_models()
            best_score = avg_score

        output_file.write(
            'episode: {0}, score: {1}, average score: {2:.1f}, best score: {3:.2f}, steps: {4}\n'.format(
                i, score, avg_score, best_score, n_steps))
        print('episode:', i, ', score:', score, ', average score: %.1f' % avg_score, ', best score: %.2f' % best_score,
              ', steps:', n_steps)

    x = [i + 1 for i in range(n_games)]
    plot_learning_curve(x, scores, figure_file)
