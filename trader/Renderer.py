import numpy as np
import matplotlib.pyplot as plt


class Renderer:
    def __init__(self, window_size=100, profit_limit=2000) -> None:
        self.figure, self.axes = plt.subplots(2, 1, figsize=(10, 8))
        self.figure.suptitle(
            'Stock Market Simulation')
        self.figure.subplots_adjust(hspace=0.5)

        self.window_size = window_size

        # profit
        self.axes[0].set_ylabel("Profit")
        self.axes[0].set_xlabel("Timestep")
        self.profit_line = self.axes[0].plot(
            [0, 1, 2], [0, 100, 50], color='blue')[0]
        self.profit_x = []
        self.profit_y = []
        self.profit_limit = profit_limit
        self.axes[0].set_ylim(-self.profit_limit, self.profit_limit)
        self.axes[0].set_xlim(0, 100)

        # price
        self.axes[1].set_ylabel("Price")
        self.axes[1].set_xlabel("Timestep")
        self.price_line = self.axes[1].plot(
            [0, 1, 2], [0, 100, 50], color='black')[0]
        self.price_x = []
        self.price_y = []

        # action
        self.action_scatter = self.axes[1].scatter(
            [0, 1, 2], [0, 100, 50], s=[100, 100, 100], color=["orange"])
        self.action_x = []
        self.action_y = []
        self.action_sizes = []
        self.action_colors = []

        # draw
        self.bg = self.figure.canvas.copy_from_bbox(self.figure.bbox)
        self.figure.draw_artist(self.price_line)
        self.figure.draw_artist(self.profit_line)
        self.figure.draw_artist(self.action_scatter)

        self.figure.canvas.blit(self.figure.bbox)
        self.figure.canvas.flush_events()

    def show(self):
        plt.show(block=False)

    def draw_frame(self):
        self.axes[0].set_xlim(
            self.profit_x[-1] - self.window_size, self.profit_x[-1]
        )
        self.axes[1].set_xlim(
            self.price_x[-1] - self.window_size, self.price_x[-1]
        )

    def pause(self, interval):
        plt.pause(interval)

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
        for action in self.action_y:
            if action > 0:
                self.action_colors.append("green")
                self.action_sizes.append(action * 10)
            elif action < 0:
                self.action_colors.append("red")
                self.action_sizes.append(-action * 10)
            else:
                self.action_colors.append("orange")
                self.action_sizes.append(10)

    def draw_action(self):
        self.action_scatter.set_offsets(np.c_[self.action_x, self.price_y])
        self.action_scatter.set_sizes(self.action_sizes)
        self.action_scatter.set_color(self.action_colors)
        self.figure.draw_artist(self.action_scatter)
