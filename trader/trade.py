import gymnasium as gym
import pandas as pd
from StockMarketEnv import StockMarketEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from Schedule import linear_schedule
from tensorboardX import SummaryWriter
import time
import os

# Create a TensorBoard writer


def train(data: pd.DataFrame, total_timesteps=1_000_000):
    print("Training Begins...")

    learning_rate_begin = 0.01
    learning_rate_end = 0.0001
    gamma = 0.99
    n_steps = 100
    batch_size = 256
    clip_range = 0.2
    window_size = 336
    initial_balance = 10000
    timestamp = time.strftime('%Y%m%d%H%M%S', time.localtime())

    os.makedirs("./trader/results/", exist_ok=True)
    os.makedirs(f"./trader/results/{timestamp}", exist_ok=True)

    # create a txt file to store the settings
    with open(f"./trader/results/{timestamp}/settings.txt", "w") as f:
        f.write(f"learning_rate_begin: {learning_rate_begin}\n")
        f.write(f"learning_rate_end: {learning_rate_end}\n")
        f.write(f"gamma: {gamma}\n")
        f.write(f"n_steps: {n_steps}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"clip_range: {clip_range}\n")
        f.write(f"window_size: {window_size}\n")
        f.write(f"total_timesteps: {total_timesteps}\n")
        f.write(f"initial_balance: {initial_balance}\n")

    env = StockMarketEnv(data, initial_balance=10000,
                         verbose=0, window_size=336)
    check_env(env, warn=True)

    model = PPO("MlpPolicy", env, verbose=1, learning_rate=linear_schedule(0.01, 0.0001), gamma=0.99,
                tensorboard_log="./trader/tensorboard/", n_steps=100, batch_size=256, clip_range=0.2)

    model.learn(total_timesteps=total_timesteps)
    print("Saving model...")

    model.save("./trader/results/" + timestamp + "/trade.pt")

    return timestamp


def test(data: pd.DataFrame, timestamp: str):
    env = StockMarketEnv(data, initial_balance=10000,
                         verbose=1, window_size=336)
    model = PPO.load("./trader/results/" + timestamp + "/trade.pt", env=env)
    vec_env = model.get_env()
    vec_env.render_mode = 'human'
    obs = vec_env.reset()

    print("Testing...")
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render()


def main():
    data_path = 'trader/dataset/AAPL.csv'
    data = pd.read_csv(data_path)
    data.dropna(inplace=True)

    # divide into train and test
    train_data = data.iloc[:int(0.8 * len(data))]
    train_data = train_data.reset_index(drop=True)

    test_data = data.iloc[int(0.8 * len(data)):]
    test_data = test_data.reset_index(drop=True)

    # timestamp = train(train_data)
    timestamp = "20240411044235"
    test(test_data, timestamp)


main()
