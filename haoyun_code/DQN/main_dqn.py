import gym
import numpy as np
from dqn_agent import DQNAgent
from utils import plot_learning_curve, make_env
from gym import wrappers
import time
import sys
# adding Folder_2 to the system path
sys.path.insert(0, r'/home/elvinzhu/github/shooter-squad/haoyun_code')
from Env import *


if __name__ == '__main__':
    env_name = 'PongNoFrameskip-v4'
    env_name = 'CartPole-v1'
    env_name = 'Breakout-v0'
    env_name = 'moba_v0'
    env_name = 'shooter'
    env = make_env(env_name)
    #env = gym.make('CartPole-v1')
    best_score = -np.inf
    load_checkpoint = False
    # n_games = 400
    n_games = 4000

    agent = DQNAgent(gamma=0.99, epsilon=1, lr=0.0001,
                     input_dims=(env.observation_space.shape),
                     n_actions=env.action_space.n, mem_size=30000, eps_min=0.1,
                     batch_size=64, replace=1000, eps_dec=1e-5,
                     chkpt_dir='models/', algo='DQNAgent',
                     env_name=env_name)

    if load_checkpoint:
        agent.load_models()

    fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) +'_' \
            + str(n_games) + 'games'
    
    # if you want to record video of your agent playing, do a mkdir tmp && mkdir tmp/dqn-video
    # and uncomment the following 2 lines.
    #env = wrappers.Monitor(env, "tmp/dqn-video",
    #                    video_callable=lambda episode_id: True, force=True)
    n_steps = 0
    scores, eps_history, steps_array = [], [], []

    time_prev = time.time()

    for i in range(n_games):
        done = False
        observation = env.reset()

        score = 0
        while not done:
            # // NOTE: whenever the agent makes a move, he enters a new state. it store the observation, action, reward, obseravation_, done in his memory for REPLAY, which has size of 40000.
            action = agent.choose_action(observation)  
            observation_, reward, done, info = env.step(action)
            # print(observation.shape)
            score += reward

            if not load_checkpoint: # NOTE: obseration: [32, 4, 84, 84]; actions: [32]; obseration_: [32, 4, 84, 84]; dones: [32]; rewards:[32]
                agent.store_transition(observation, action,
                                     reward, observation_, done)
                
                agent.learn()
            observation = observation_
            n_steps += 1
            
        scores.append(score)
        steps_array.append(n_steps)

        avg_score = np.mean(scores[-100:])
        print('episode: ', i,'score: ', score,
             ' average score %.1f' % avg_score, 'best score %.2f' % best_score,
            'epsilon %.2f' % agent.epsilon, 'steps', n_steps, 'time', time.time()-time_prev)
        time_prev = time.time()

        if avg_score > best_score:
            # if not load_checkpoint:
            #     agent.save_models()
            best_score = avg_score

        eps_history.append(agent.epsilon)

        if i % 100 == 10:
            agent.save_models()
            figure_file = 'plots/' + fname +'_'+str(i)+'_.png'
            plot_learning_curve(steps_array, scores, eps_history, figure_file)

    figure_file = 'plots/' + fname + '_final.png'
    plot_learning_curve(steps_array, scores, eps_history, figure_file)
