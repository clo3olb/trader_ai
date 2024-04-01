from stable_baselines3 import PPO
from trader.StockMarketEnv import StockEnv


class StockMarketAgent:
    def __init__(self, env: StockEnv, stocks: list[str]):
        self.stocks = stocks
        self.model = PPO("MlpPolicy", env, verbose=1)

    def train(self, steps: int):
        self.model.learn(total_timesteps=steps)

    def save(self, path: str):
        self.model.save(path)

    def load(self, path: str):
        self.model.load(path)


# Example usage
stocks = ["AAPL", "MSFT", "AMZN"]
agent = StockMarketAgent(stocks)
agent.train(10000)
