import gym
import numpy as np
from dqn_agent import DQNAgent
from Env.utils import plot_learning_curve, make_env
from gym import wrappers
import time
import sys
# adding Folder_2 to the system path
# sys.path.insert(0, r'../')
from Env import *


if __name__ == '__main__':
    env_name = 'shooter'
    env = make_env(env_name)

    best_score = -np.inf
    load_checkpoint = False
    n_games = 4000

    epsilon = 1.0
    if load_checkpoint:
        epsilon = 0.0


    # print('env obervation space shape is ', env.observation_space.shape)
    agent = DQNAgent(gamma=0.99, epsilon=0, lr=0.0001,
                     input_dims=(env.observation_space.shape),
                     n_actions=env.action_space.n, mem_size=30000, eps_min=0.1,
                     batch_size=64, replace=1000, eps_dec=1e-5,
                     chkpt_dir='models/', algo='Env_v1.5_DQNAgent',
                     env_name=env_name)

    if load_checkpoint:
        agent.load_models()

    fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) +'_' \
            + str(n_games) + 'games'
    figure_file = 'plots/' + fname + '.png'
    # if you want to record video of your agent playing, do a mkdir tmp && mkdir tmp/dqn-video
    # and uncomment the following 2 lines.
    #env = wrappers.Monitor(env, "tmp/dqn-video",
    #                    video_callable=lambda episode_id: True, force=True)
    n_steps = 0
    scores, eps_history, steps_array = [], [], []

    time_prev = 0
    time_curr = 0

    output_file = open("dqn_env_v1.5.txt", "w")
    accuracy_file = open('accuracy_image_neg', 'w')
    for i in range(n_games):
        done = False
        observation = env.reset()

        score = 0
        while not done:
            # // NOTE: whenever the agent makes a move, he enters a new state. it store the observation, action, reward, obseravation_, done in his memory for REPLAY, which has size of 40000.
            action = agent.choose_action(observation) # * action shape is scalar (e.g. 3)
            
            observation_, reward, done, info = env.step(action) # * observation shape is (4, 84, 84), reward = scalar, all variables are unbatched
            # print(observation_.shape, reward)
            time_prev = time.time()
            score += reward

            if not load_checkpoint:
                agent.store_transition(observation, action,
                                     reward, observation_, done)
                
                agent.learn()
            observation = observation_
            n_steps += 1
            time_curr = time.time()

        scores.append(score)
        steps_array.append(n_steps)

        avg_score = np.mean(scores[-100:])

        output_file.write('episode: {0}, score: {1}, average score: {2:.1f}, best score: {3:.2f}, epsilon: {4:.2f}, steps: {5}\n'.format(i, score, avg_score, best_score,agent.epsilon, n_steps))

        print('episode: ', i,', score: ', score,
                ', average score: %.1f' % avg_score, ', best score: %.2f' % best_score,
                ', epsilon: %.2f' % agent.epsilon, ', steps: ', n_steps)
        accuracy_file.write('episode: ' + str(i) + ' score: ' + str(score) +
                ' average score: ' + str(avg_score) + ' best score: ' + str(best_score) +
                ' epsilon: ' + str(agent.epsilon) + ' steps: ' + str(n_steps) + '\n')
        if avg_score > best_score:
            # if not load_checkpoint:
            #     agent.save_models()
            best_score = avg_score

        eps_history.append(agent.epsilon)

        if i % 50 == 49:
            agent.save_models()
    accuracy_file.close()
    x = [i + 1 for i in range(len(scores))]
    plot_learning_curve(steps_array, scores, eps_history, figure_file)
