from StockMarketEnv import StockMarketEnv
import pandas as pd
import time

data = pd.read_csv('trader/dataset/AAPL.csv')
data = data.dropna()


env = StockMarketEnv(data, verbose=2)
env.reset()

for i in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render(interval=0.5)
    if terminated or truncated:
        env.reset()
