
import torch
import random

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import gymnasium as gym
from gymnasium import Env

from stable_baselines3 import DDPG
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.ddpg.policies import MlpPolicy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import VecEnv


class DQNAgent:
    def __init__(self, env: Env, state_size, action_size, epsilon=1.0, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99  # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99995
        self.env = env
        new_logger = configure("trader/log", ["stdout", "csv"])
        self.model: nn.Module = DDPG(
            MlpPolicy,
            env,
            learning_rate,
            buffer_size=10000,
            learning_starts=100,
            batch_size=1,
            tau=0.005,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            optimize_memory_usage=False,
            tensorboard_log=None,
            policy_kwargs=None,
            verbose=0,
            seed=None,
            device='auto',
            replay_buffer_class=ReplayBuffer,
            replay_buffer_kwargs=None,
            _init_setup_model=True
        )
        self.model.set_logger(new_logger)
        self.vec_env = VecEnv([env])

    def remember(self, state, action, reward, next_state, dones):
        self.model.collect_rollouts(
            self.vec_env, callback=None, train_freq=4, replay_buffer=self.model.replay_buffer)

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.random() * self.action_size
        else:
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            q_values = self.model(state)
            return q_values.max(1)[1].item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(
            *batch)
        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float)
        dones = torch.tensor(dones, dtype=torch.float).unsqueeze(1)

        self.model.train()
        next_q_values = self.model(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = F.smooth_l1_loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
