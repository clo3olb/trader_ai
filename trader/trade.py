import gymnasium as gym
import pandas as pd
from StockMarketEnv import StockMarketEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


data_path = 'trader/dataset/AAPL.csv'
data = pd.read_csv(data_path)
data = data.dropna()
# data = data.head(100)

env = StockMarketEnv(data)
check_env(env, warn=True)

model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.001)
model.learn(total_timesteps=10000)

print("Saving model...")
model.save("./trader/trade.pt")

mean_reward, std_reward = evaluate_policy(
    model, model.get_env(), n_eval_episodes=10)

vec_env = model.get_env()
vec_env.render_mode = 'human'
obs = vec_env.reset()

print("Rendering...")
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render()


# model = PPO.load("./trader/lunar_project/lunar.pt", env=env)
# model.save("./trader/lunar_project/lunar.pt")
