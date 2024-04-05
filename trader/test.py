from StockMarketEnv import StockMarketEnv
import pandas as pd
import time

data = pd.read_csv('trader/dataset/AAPL.csv')
data = data.dropna()


env = StockMarketEnv(data, verbose=1, initial_balance=10000)
env.reset()

for i in range(3):
    action = -1
    obs, reward, terminated, truncated, info = env.step(action)
    env.render(interval=0.1)
    if terminated or truncated:
        env.reset()
