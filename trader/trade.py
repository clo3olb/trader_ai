import gymnasium as gym
import pandas as pd
from StockMarketEnv import StockMarketEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from Schedule import linear_schedule
from TensorboardCallback import TensorBoardCallback
from tensorboardX import SummaryWriter

# Create a TensorBoard writer


def train(data: pd.DataFrame, total_timesteps=100_000):
    print("Training Begins...")
    env = StockMarketEnv(data, initial_balance=10000, verbose=0)
    check_env(env, warn=True)

    model = PPO("MlpPolicy", env, verbose=1, learning_rate=linear_schedule(0.01, 0.0001), gamma=0.99,
                tensorboard_log="./trader/tensorboard/", n_steps=100, batch_size=256, clip_range=0.2)

    writer = SummaryWriter("./trader/tensorboard/")
    model.learn(total_timesteps=total_timesteps,
                callback=TensorBoardCallback(writer))
    print("Saving model...")
    model.save("./trader/trade.pt")


def test(data: pd.DataFrame):
    env = StockMarketEnv(data, initial_balance=10000, verbose=1)
    model = PPO.load("./trader/trade.pt", env=env)
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

    train(train_data)
    test(test_data)


main()
