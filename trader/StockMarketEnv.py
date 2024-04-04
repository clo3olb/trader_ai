# From https://medium.com/@sthanikamsanthosh1994/custom-gym-environment-stock-trading-for-reinforcement-learning-stable-baseline3-629a489d462d
import time
import random
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
from Renderer import Renderer

INITIAL_ACCOUNT_BALANCE = 10000
MAX_ACCOUNT_BALANCE = INITIAL_ACCOUNT_BALANCE * 10
MAX_NUM_SHARES = 7421640800
MAX_SHARE_PRICE = 5000


class StockMarketEnv(gym.Env):
    def __init__(self, df: pd.DataFrame, initial_balance=INITIAL_ACCOUNT_BALANCE, verbose=0):
        super(StockMarketEnv, self).__init__()

        self.initial_balance = initial_balance
        self.verbose = verbose

        self.df = df

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

        # if any of the value larger than 1
        for row in range(len(obs)):
            for col in range(len(obs[row])):
                if obs[row][col] > 1:
                    print("Balance: ", self.balance)
                    print(obs)
                    raise ValueError(
                        f"Value at row {row} and col {col} is greater than 1, and timestep is {self.current_step}")

        return obs.astype(np.float32)

    def _get_current_price(self):
        return random.uniform(self.df.loc[self.current_step, "Open"], self.df.loc[self.current_step, "Close"])

    def _take_action(self, action):
        # Set the current price to a random price within the time step
        current_price = self._get_current_price()

        self.price_history.append(current_price)
        action_type = action[0]
        amount = action[1]

        if action_type > 0:
            if amount >= 0:
                # Buy amount % of balance in shares
                if self.balance == 0:
                    self.action_history.append(0)
                    return -10

                total_possible = int(self.balance / current_price)
                shares_bought = int(total_possible * amount)
                prev_cost = self.cost_basis * self.shares_held
                additional_cost = shares_bought * current_price

                self.balance -= additional_cost
                self.cost_basis = (
                    prev_cost + additional_cost) / (self.shares_held + shares_bought)
                self.shares_held += shares_bought
                self.net_worth = self.balance + self.shares_held * current_price
                self.action_history.append(shares_bought)

            elif amount < 0:
                # Sell amount % of shares held
                if self.shares_held == 0:
                    self.action_history.append(0)
                    return -10

                shares_sold = int(self.shares_held * -amount)
                self.balance += shares_sold * current_price
                self.shares_held -= shares_sold
                self.total_shares_sold += shares_sold
                self.total_sales_value += shares_sold * current_price
                self.net_worth = self.balance + self.shares_held * current_price
                self.action_history.append(-shares_sold)
        else:
            self.action_history.append(0)
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0

    def step(self, action):
        # Execute one time step within the environment
        # clear the terminal

        reward = 0

        self.timesteps += 1
        self._take_action(action)
        self.current_step += 1
        profit = self.net_worth - self.initial_balance
        self.profit_history.append(profit)

        if self.verbose == 1:
            self.info("\033[H\033[J")
            self.info(f"Step: {self.timesteps}")
            self.info(f"Action: {action}")
            self.info(f"Profit: {profit}")
            self.info(f"Net worth: {self.net_worth}")
            self.info(f"Max net worth: {self.max_net_worth}")
            self.info(f"Shares held: {self.shares_held}")
            self.info(f"Cost basis: {self.cost_basis}")
            self.info(f"Total shares sold: {self.total_shares_sold}")
            self.info(f"Reward: {reward}")

        if abs(profit) > self.initial_balance * self.worth_change_limit:
            reward += profit

        terminated: bool = bool(abs(
            profit) > self.initial_balance * self.worth_change_limit)
        self.info(f"Terminated: {terminated}")
        truncated: bool = self.current_step >= len(
            self.df.loc[:, 'Open'].values)
        self.info(f"Truncated: {truncated}")

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
        self.worth_change_limit = 0.2
        self.profit_history = []
        self.price_history = []
        self.action_history = []
        self.timesteps = 0

        # render
        self.renderer = None

        # Set the current step to a random point within the data frame
        self.current_step = random.randint(
            6, len(self.df['Open'].values) - 6)

        self.debug("Current step reset:", self.current_step)

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

    def render(self, mode='human', close=False, interval=0.1):
        if self.renderer is None:
            self.renderer = Renderer(
                window_size=100, profit_limit=self.initial_balance * self.worth_change_limit)
            self.renderer.show()
        else:
            self.renderer.update_profit(self.profit_history)
            self.renderer.update_price(self.price_history)
            self.renderer.update_action(self.action_history)

            self.renderer.draw_frame()
            self.renderer.draw_profit()
            self.renderer.draw_price()
            self.renderer.draw_action()

            self.renderer.pause(interval)

    def _log(self, args, type="info"):
        if self.verbose == 1 and type == "info":
            print(args)
        if self.verbose == 2 and type == "debug":
            print(args)

    def info(self, *args):
        self._log(args, "info")

    def debug(self, *args):
        self._log(args, "debug")
