import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class Renderer:
    def __init__(self, window_size=100, profit_limit=2000, initial_balance=1000) -> None:
        self.figure, self.axes = plt.subplots(3, 1, figsize=(10, 8))
        self.figure.suptitle(
            'Stock Market Simulation')
        self.figure.subplots_adjust(hspace=0.2)

        self.window_size = window_size
        self.initial_balance = initial_balance
        self.profit_limit = profit_limit

        # profit
        self.axes[0].set_ylabel("Profit")
        self.profit_line = self.axes[0].plot(
            [0, 1, 2], [0, 100, 50], color='blue')[0]
        self.profit_x = []
        self.profit_y = []
        self.axes[0].set_ylim(-self.profit_limit, self.profit_limit)
        self.axes[0].set_xlim(0, 100)

        # price
        self.axes[1].set_ylabel("Price")
        self.price_line = self.axes[1].plot(
            [0, 1, 2], [0, 100, 50], color='black')[0]
        self.price_x = []
        self.price_y = []

        # action
        self.action_scatter = self.axes[1].scatter(
            [0, 1, 2], [0, 100, 50], s=[100, 100, 100], color=["orange", "red", "green"])
        self.action_x = []
        self.action_y = []
        self.action_sizes = []
        self.action_colors = []
        # produce a legend with the unique colors from the scatter
        legend1 = self.axes[1].legend(*self.action_scatter.legend_elements(),
                                      loc="lower left", title="Classes")
        self.axes[1].add_artist(legend1)

        # portfolio
        self.axes[2].set_ylabel("Portfolio")
        self.portfolio_values = pd.DataFrame(
            data=[], columns=["Cash", "Shares"])

        # draw
        self.bg = self.figure.canvas.copy_from_bbox(self.figure.bbox)

        self.figure.draw_artist(self.price_line)
        self.figure.draw_artist(self.profit_line)
        self.figure.draw_artist(self.action_scatter)
        # self.figure.draw_artist(self.portfolio_bars)

        self.figure.canvas.blit(self.figure.bbox)
        self.figure.canvas.flush_events()

    def show(self):
        plt.show(block=False)

    def draw_frame(self):
        if len(self.profit_y) > 0:
            self.axes[0].set_xlim(
                self.profit_x[-1] - self.window_size, self.profit_x[-1]
            )
            max_profit = min(max([abs(profit)
                                  for profit in self.profit_y] + [1]), self.profit_limit) * 1.2
            self.axes[0].set_ylim(-max_profit, max_profit)
        else:
            self.axes[0].set_ylim(-self.profit_limit, self.profit_limit)

        if len(self.price_y) > 0:
            self.axes[1].set_xlim(
                self.price_x[-1] - self.window_size, self.price_x[-1]
            )
            self.axes[1].set_ylim(
                min(self.price_y) *
                0.2, max(self.price_y) * 1.2
            )
        else:
            self.axes[1].set_ylim(0, 100)

        if len(self.action_y) > 0:
            self.axes[1].set_xlim(
                self.action_x[-1] - self.window_size, self.action_x[-1]
            )

    def pause(self, interval):
        plt.pause(interval)

    def update_balance(self, balance):
        self.axes[0].set_title(f"Balance: {balance}")

    def update_profit(self, profit_history):
        self.profit_x = np.arange(
            len(profit_history) - self.window_size, len(profit_history)
        )
        if len(profit_history) < self.window_size:
            self.profit_x = np.arange(0, len(profit_history))
        self.profit_y = profit_history[-self.window_size:]

    def draw_profit(self):
        self.profit_line.set_xdata(self.profit_x)
        self.profit_line.set_ydata(self.profit_y)
        self.figure.draw_artist(self.profit_line)

    def update_price(self, price_history):
        self.price_x = np.arange(
            len(price_history) - self.window_size, len(price_history)
        )
        if len(price_history) < self.window_size:
            self.price_x = np.arange(0, len(price_history))
        self.price_y = price_history[-self.window_size:]

    def draw_price(self):
        self.price_line.set_xdata(self.price_x)
        self.price_line.set_ydata(self.price_y)
        self.figure.draw_artist(self.price_line)

    def update_action(self, action_history):
        self.action_x = np.arange(
            len(action_history) - self.window_size, len(action_history)
        )
        if len(action_history) < self.window_size:
            self.action_x = np.arange(0, len(action_history))
        self.action_y = action_history[-self.window_size:]
        self.action_sizes = []
        self.action_colors = []
        dot_size = 20
        for action in self.action_y:
            if action > 0:
                self.action_colors.append("green")
                self.action_sizes.append(20)
            elif action < 0:
                self.action_colors.append("red")
                self.action_sizes.append(20)
            else:
                self.action_colors.append("white")
                self.action_sizes.append(20)

    def draw_action(self):
        self.action_scatter.set_offsets(np.c_[self.action_x, self.price_y])
        self.action_scatter.set_sizes(self.action_sizes)
        self.action_scatter.set_color(self.action_colors)
        self.figure.draw_artist(self.action_scatter)

    def update_portfolio(self, cash, shares):
        # pad with zeros if the dataframe is empty
        for _ in range(len(self.portfolio_values.index), self.window_size):
            self.portfolio_values.loc[len(self.portfolio_values.index)] = [
                0, 0
            ]
        # reset index
        self.portfolio_values.index = np.arange(self.window_size)
        self.portfolio_values.loc[len(self.portfolio_values.index)] = [
            cash, shares]

        if len(self.portfolio_values.index) > self.window_size:
            self.portfolio_values = self.portfolio_values.iloc[-100:]

    def draw_portfolio(self):
        self.axes[2].cla()
        self.portfolio_values.plot(
            kind='bar',
            stacked=True,
            colormap='tab10',
            ax=self.axes[2],
            width=0.9
        )
        self.axes[2].axhline(y=self.initial_balance,
                             color='red', linestyle='--')
        self.axes[2].get_xaxis().set_ticks([])

    def close(self):
        # remove all
        plt.close()
