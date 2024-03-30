import gymnasium as gym
import pandas as pd
import matplotlib.pyplot as plt

from Account import Account


class StockEnv(gym.Env):
    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance=10000,
        trading_fee=0.01,
        window_size=10,
        date_header="Date",
        target_headers=["Close"],
    ):
        self.data: pd.DataFrame = data.drop(columns=["Date"])

        self.initial_balance = initial_balance
        self.trading_fee = trading_fee

        self.action_space = gym.spaces.Box(low=[0], high=[1], dtype=float)
        self.observation_space = gym.spaces.Box(
            low=0, high=100000, shape=(len(self._getStates(0)),)
        )

        self.reset()

    def reset(self):
        self.account = Account(balance=self.initial_balance,
                               trading_fee=self.trading_fee)
        self.shares = 0
        self.current_step = 0
        self.done = False
        self.history = []
        return self._getStates(self.current_step)

    def _buy(self, amount: float):
        if self.balance >= amount:
            print("Buying: ", amount, "Price: ",
                  self._getPrice(self.current_step))
            change_in_shares = amount / self._getPrice(self.current_step)
            change_in_balance = -amount

            self.shares += change_in_shares
            self.balance += change_in_balance
            return 0
        else:
            return -10

    def _sell(self, amount: float):
        if self.shares * self._getPrice(self.current_step) >= amount:
            print("Selling: ", amount, "Price: ",
                  self._getPrice(self.current_step))
            change_in_shares = -amount / self._getPrice(self.current_step)
            change_in_balance = amount

            self.balance += change_in_balance
            self.shares += change_in_shares
            return 0
        else:
            return -10

    def step(self, action):
        if self.done:
            return self._getStates(self.current_step), 0, self.done, {
                'balance': self.balance,
                'shares': self.shares,
                'value': self._getValue()
            }
        if action < 0 or action > 1:
            raise ValueError('Action must be between 0 and 1')
        self.current_step += 1

        if self.current_step >= len(self.data) - 1:
            self.done = True

        reward = 0

        # need to have action(0~1) amount of shares
        value = self._getPrice(self.current_step)
        amount_change = action * value - self.shares * \
            self._getPrice(self.current_step)

        if amount_change > 0:
            reward = self._buy(amount_change)
        elif amount_change < 0:
            reward = self._sell(-amount_change)

        if self.done:
            self.balance = self.balance + self.shares * \
                self._getPrice(self.current_step)
            reward = self.balance - self.initial_balance

        self.history.append(self._getValue(self.current_step))

        return self._getStates(self.current_step), reward, self.done, {
            'balance': self.balance,
            'shares': self.shares,
            'value': self._getValue(self.current_step)
        }

    def render(self):
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(self.data['Close'])
        ax2.plot(self.history, color='r')
        ax1.set_ylabel('Close Price')
        ax2.set_ylabel('Transaction Amount')
        plt.savefig('trader/plot.png')
