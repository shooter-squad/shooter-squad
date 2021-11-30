import numpy as np


class ObservationSampler:
    """
    Sample observation as CAM extractor input and output
    """
    def __init__(self, env):
        self.env = env
        self.observation = self.env.reset()

    def sample_observation(self, n_frames_after):
        # Randomly perform action until n_frames_after
        for _ in range(n_frames_after):
            action = np.random.choice(self.env.action_space.n)
            self.observation, _, done, _ = self.env.step(action)

            if done:
                self.observation = self.env.reset()

        return self.observation
