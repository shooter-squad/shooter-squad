import time

from Env import *
from Env.utils import plot_learning_curve, make_env
from dqn_agent import DQNAgent

if __name__ == '__main__':
    # -------------------- Parameters begin --------------------
    env_name = 'shooter'
    best_score = -np.inf
    load_checkpoint = False
    n_games = 3000
    pre_train = False
    step_limit_per_game = 2500

    if load_checkpoint:
        epsilon = 0.0
        eps_min = 0.001
    elif pre_train:
        epsilon = 0.5
        eps_min = 0.1
    else:
        epsilon = 1.0
        eps_min = 0.001
    # -------------------- Parameters end --------------------

    env = make_env(env_name)

    agent = DQNAgent(gamma=0.99, epsilon=epsilon, lr=0.0001,
                     input_dims=(env.observation_space.shape),
                     n_actions=env.action_space.n, mem_size=30000, eps_min=eps_min,
                     batch_size=64, replace=1000, eps_dec=1e-5,
                     chkpt_dir='models/', algo='DQNAgentScratch',
                     env_name=env_name)

    if load_checkpoint:
        agent.load_models()

    if pre_train:
        agent.PreTrain()

    fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) + '_' \
            + str(n_games) + 'games'
    figure_file = 'plots/' + fname + '.png'
    # if you want to record video of your agent playing, do a mkdir tmp && mkdir tmp/dqn-video
    # and uncomment the following 2 lines.
    # env = wrappers.Monitor(env, "tmp/dqn-video",
    #                    video_callable=lambda episode_id: True, force=True)
    n_steps = 0
    scores, eps_history, steps_array = [], [], []

    time_prev = 0
    time_curr = 0

    output_file = open("stats_dqn_scratch.txt", "w")
    output_file.close()
    accuracy_file = open('accuracy_image_neg', 'w')
    for i in range(n_games):
        done = False
        observation = env.reset()
        info_stack = env.get_info_stack()
        score = 0
        cur_step = 0
        while not done and cur_step < step_limit_per_game:
            # // NOTE: whenever the agent makes a move, he enters a new state. it store the observation, action, reward, obseravation_, done in his memory for REPLAY, which has size of 40000.
            # print('ENTERING THE GAME')

            action = agent.choose_action(observation, info_stack)  # * action shape is scalar (e.g. 3)
            # print('observation shape ', observation.shape)
            # print(info_stack)
            # print('action shape is: ', action)
            # print('in one iteration')

            # fire_file = open('fire.txt', 'a')
            # if action == 3:
            #     fire_file.write(str(action))
            #     print('fire')
            # else:
            #     fire_file.write(str('_'))
            #     print('None')
            # fire_file.close()

            observation_, reward, done, info = env.step(
                action)  # * observation shape is (4, 84, 84), reward = scalar, all variables are unbatched, info-stack: (4, 6)
            info_stack_ = env.get_info_stack()
            # print('INFO_STACK')
            # print(info_stack)
            # print('action shape: ', action, 'obseravation shape', observation.shape)
            time_prev = time.time()
            score += reward

            if not load_checkpoint:
                agent.store_transition(observation, action,
                                       reward, observation_, done, info_stack, info_stack_)

                agent.learn()
            observation = observation_
            info_stack = info_stack_
            cur_step += 1
            time_curr = time.time()
            # print('DONE: ', done)

        n_steps += cur_step
        scores.append(score)
        steps_array.append(n_steps)

        avg_score = np.mean(scores[-100:])

        output_file = open("stats_dqn_scratch.txt", "a")
        output_file.write(
            'episode: {0}, score: {1}, average score: {2:.1f}, best score: {3:.2f}, epsilon: {4:.2f}, steps: {5}\n'.format(
                i, score, avg_score, best_score, agent.epsilon, n_steps))
        output_file.close()

        print('episode: ', i, ', score: ', score,
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
