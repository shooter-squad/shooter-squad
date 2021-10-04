from gym import Env
from gym.spaces import Discrete

from game_scene import GameScene


class ShooterEnv(Env):
    """
    The custom environment class for our shooter game.
    """

    def __init__(self):
        self.game_scene = GameScene()
        self.action_space = Discrete(self.game_scene.ActionCount())
        self.observation_space = self.game_scene.ScreenShot()
        self.state = self.game_scene.ScreenShot()

        self.reward = 0
        self.done = self.game_scene.Done()
        self.info = {}

    def step(self, action_num: int):
        self.game_scene.Play(action_num)
        self.reward = self.game_scene.Reward()
        self.state = self.game_scene.ScreenShot()
        self.done = self.game_scene.Done()
        self.info = {}

        return self.state, self.reward, self.done, self.info

    def render(self, mode="human"):
        pass

    def reset(self):
        self.state = self.game_scene.Reset()
        return self.state


if __name__ == '__main__':
    env = ShooterEnv()

    action_list = [
        0, 1, 2, 3, 0, 0, 1, 2, 3, 0, 0, 1, 2, 3, 0, 0, 1, 2, 3, 0, 0, 1, 2, 3, 0
    ]

    for action in action_list:
        env.step(action)
        env.step(action)
        env.step(action)
        env.step(action)
        env.step(action)
