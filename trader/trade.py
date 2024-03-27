import pandas as pd
from StockEnv import StockEnv
from DQN import DQNAgent

data_path = 'trader/dataset/AAPL.csv'
data = pd.read_csv(data_path)
data = data.head(100)


batch_size = 1


def train(episodes: int = 100, initial_balance=1000, batch_size: int = 1, learning_rate: float = 0.001, epsilon: float = 1.0, load: bool = False):
    env = StockEnv(data, initial_balance)
    agent = DQNAgent(env, state_size=env.getStateSize(),
                     action_size=env.getActionSize(), learning_rate=learning_rate, epsilon=epsilon)
    if load:
        agent.load('trader/model.pth')

    for episode in range(episodes):
        print('Episode:', episode + 1)
        obs = env.reset()
        count = 0
        while True:
            count += 1

            action = agent.act(obs)

            next_obs, reward, done, info = env.step(action)
            agent.remember(obs, action, reward, next_obs, done)
            obs = next_obs

            loss = agent.replay(batch_size=batch_size)
            if done:
                print(info)
                break

    agent.save('trader/model.pth')
    env.render()
    print(env.history)


def test(initial_balance=1000):
    agent = DQNAgent(state_size=env.getStateSize(),
                     action_size=env.getActionSize(), epsilon=0)
    agent.load('trader/model.pth')
    env = StockEnv(data, initial_balance)

    obs = env.reset()
    while True:
        action = agent.act(obs)
        next_obs, reward, done, info = env.step(action)
        obs = next_obs

        if done:
            print(info)
            break


train(episodes=10, initial_balance=1000,  batch_size=1,
      learning_rate=0.001, epsilon=1.0, load=False)
# test()
