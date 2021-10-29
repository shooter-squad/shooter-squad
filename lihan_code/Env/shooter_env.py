import numpy as np
from gym import Env
from gym.spaces import Discrete, Box

from Env.constants import *
from Env.game_scene import GameScene


class ShooterEnv(Env):
    """
    The custom environment class for our shooter game.
    """

    def __init__(self):
        self.game_scene = GameScene()
        self.action_space = Discrete(self.game_scene.ActionCount())
        # self.observation_space = self.game_scene.ScreenShot()
        self.observation_shape = (WIDTH, HEIGHT, 3)
        self.observation_space = Box(low=np.zeros(self.observation_shape),
                                     high=np.full(self.observation_shape, 255),
                                     dtype=np.uint8)
        self.state = self.game_scene.ScreenShot()

        self.reward = 0
        self.done = self.game_scene.Done()
        self.info = {}
        self.additional_state = self.game_scene.AdditionalState()

    def step(self, action_num: int):
        # More return values
        self.done = self.game_scene.Play(action_num)
        self.reward = self.game_scene.Reward()
        self.state = self.game_scene.ScreenShot()
        self.info = {}
        self.additional_state = self.game_scene.AdditionalState()

        return self.state, self.reward, self.done, self.info, self.additional_state

    def render(self, mode="human"):
        pass

    def reset(self):
        self.game_scene.Reset()
        self.state = self.game_scene.ScreenShot()
        self.additional_state = self.game_scene.AdditionalState()
        return self.state, self.additional_state


if __name__ == '__main__':
    env = ShooterEnv()

    action_list = [
        0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7
    ]

    for action in action_list:
        state, reward, done, info, additional = env.step(action)
        print(reward, additional)
        env.step(action)
        env.step(action)
        env.step(action)
        env.step(action)
