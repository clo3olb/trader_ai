import numpy as np

from stable_baselines3.common.callbacks import BaseCallback


class TensorBoardCallback(BaseCallback):
    def __init__(self, writer):
        super(TensorBoardCallback, self).__init__()
        self.writer = writer

    def _on_step(self):
        total_reward = sum(self.locals['rewards'])
        self.writer.add_scalar(
            'Reward/Total', total_reward, self.num_timesteps)
        for i, reward in enumerate(self.locals['rewards']):
            self.writer.add_scalar(
                f'Reward/Step{i}', reward, self.num_timesteps)
        return True
