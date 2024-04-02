# From https://medium.com/@sthanikamsanthosh1994/custom-gym-environment-stock-trading-for-reinforcement-learning-stable-baseline3-629a489d462d

import random
import json
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 5000
MAX_STEPS = 20000
INITIAL_ACCOUNT_BALANCE = 10000


class StockMarketEnv(gym.Env):
    def __init__(self, df: pd.DataFrame, initial_balance=INITIAL_ACCOUNT_BALANCE):
        super(StockMarketEnv, self).__init__()

        self.initial_balance = initial_balance

        self.df = df
        self.reward_range = (np.float32(0), np.float32(MAX_ACCOUNT_BALANCE))

        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Box(
            low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)

        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(6, 6), dtype=np.float32)

    def _next_observation(self):
        # Get the stock data points for the last 5 days and scale to between 0-1
        frame = np.array([
            self.df.loc[self.current_step - 6: self.current_step -
                        1, 'Open'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step - 6: self.current_step -
                        1, 'High'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step - 6: self.current_step -
                        1, 'Low'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step - 6: self.current_step -
                        1, 'Close'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step - 6: self.current_step -
                        1, 'Volume'].values / MAX_NUM_SHARES,
        ])

        # Append additional data and scale each value to between 0-1
        obs = np.append(frame, [[
            self.balance / MAX_ACCOUNT_BALANCE,
            self.shares_held / MAX_NUM_SHARES,
            self.max_net_worth / MAX_ACCOUNT_BALANCE,
            self.cost_basis / MAX_SHARE_PRICE,
            self.total_shares_sold / MAX_NUM_SHARES,
            self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE),
        ]], axis=0)
        return obs.astype(np.float32)

    def _take_action(self, action):
        # Set the current price to a random price within the time step
        print("Begin Action: ", action)
        current_price = random.uniform(
            self.df.loc[self.current_step, "Open"], self.df.loc[self.current_step, "Close"])

        action_type = action[0]
        amount = action[1]

        if action_type > 0:
            if amount >= 0:
                # Buy amount % of balance in shares
                total_possible = int(self.balance / current_price)
                shares_bought = int(total_possible * amount)
                prev_cost = self.cost_basis * self.shares_held
                additional_cost = shares_bought * current_price

                self.balance -= additional_cost
                self.cost_basis = (
                    prev_cost + additional_cost) / (self.shares_held + shares_bought)
                self.shares_held += shares_bought

            elif amount < 0:
                # Sell amount % of shares held
                shares_sold = int(self.shares_held * amount)
                self.balance += shares_sold * current_price
                self.shares_held -= shares_sold
                self.total_shares_sold += shares_sold
                self.total_sales_value += shares_sold * current_price

        self.net_worth = self.balance + self.shares_held * current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0

        print("End Action: ", action)

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1

        delay_modifier = (self.current_step / MAX_STEPS)

        reward = self.balance * delay_modifier
        terminated: bool = bool(self.net_worth <= 0)
        truncated: bool = self.current_step >= MAX_STEPS

        obs = self._next_observation()

        info = {
            'balance': self.balance,
            'shares_held': self.shares_held,
            'net_worth': self.net_worth,
            'max_net_worth': self.max_net_worth,
            'shares_held': self.shares_held,
            'cost_basis': self.cost_basis,
            'total_shares_sold': self.total_shares_sold,
            'total_sales_value': self.total_sales_value
        }

        if self.current_step > len(self.df.loc[:, 'Open'].values) - 6:
            return obs, reward, True, truncated, info
        return obs, reward, terminated, truncated, info

    def reset(self, seed=2024):
        # Reset the state of the environment to an initial state
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0

        # Set the current step to a random point within the data frame
        self.current_step = random.randint(
            6, len(self.df.loc[:, 'Open'].values) - 6)

        observation = self._next_observation()
        return observation, {
            'balance': self.balance,
            'shares_held': self.shares_held,
            'net_worth': self.net_worth,
            'max_net_worth': self.max_net_worth,
            'shares_held': self.shares_held,
            'cost_basis': self.cost_basis,
            'total_shares_sold': self.total_shares_sold,
            'total_sales_value': self.total_sales_value
        }

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        profit = self.net_worth - self.initial_balance

        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(
            f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
        print(
            f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
        print(
            f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
        print(f'Profit: {profit}')
