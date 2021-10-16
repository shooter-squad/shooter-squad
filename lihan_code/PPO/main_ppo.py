import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
# from stable_baselines3.common.env_util import make_vec_env

from utils import plot_learning_curve, make_env

from Env import ShooterEnv

if __name__ == '__main__':
    # env_name = 'shooter'
    # env = make_env(env_name)
    env = ShooterEnv()

    # env = gym.make('CartPole-v1')

    check_env(env)

    model = PPO("CnnPolicy", env, verbose=1)
    model.learn(total_timesteps=25000)
    #
    # obs = env.reset()
    # while True:
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info = env.step(action)
    #     env.render()
