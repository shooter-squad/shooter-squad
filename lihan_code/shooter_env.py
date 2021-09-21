from gym import Env
from game import Game


class ShooterEnv(Env):
    """
    The custom environment class for our shooter game.
    """

    def __init__(self):
        self.game = Game()
        self.action_space = self.game.Actions()
        self.observation_space = self.game.ScreenShot()
        self.state = self.game.ScreenShot()

        self.reward = 0
        self.done = self.game.Done()
        self.info = {}

    def step(self, action):
        self.game.Play(action)
        self.reward = self.game.Reward()
        self.state = self.game.ScreenShot()
        self.done = self.game.Done()
        self.info = {}

        return self.state, self.reward, self.done, self.info

    def render(self, mode="human"):
        pass

    def reset(self):
        self.state = self.game.Reset()
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
