import gymnasium as gym
from trader.DQN import DQNAgent
import matplotlib.pyplot as plt

model_save_path = 'trader/lunar.pt'


def train(num_episodes=1000, load_model=False, epsilon=1.0, learning_rate=0.001):
    agent = DQNAgent(state_size=8, action_size=4,
                     epsilon=epsilon, learning_rate=learning_rate)
    if load_model:
        agent.load(model_save_path)

    env = gym.make('LunarLander-v2', render_mode='rgb_array')
    observation, info = env.reset()

    rewards = []
    losses = []

    # Train the agent
    for episode in range(num_episodes):
        print('Training Episode:', episode)
        observation, info = env.reset()

        while True:
            action = agent.act(observation)
            next_observation, reward, terminated, truncated, info = env.step(
                action)
            agent.remember(observation, action, reward,
                           next_observation, terminated, truncated)
            observation = next_observation
            loss = agent.replay(batch_size=36)
            rewards.append(reward)
            if loss is not None:
                losses.append(loss)

            if terminated or truncated:
                break

    agent.save(model_save_path)

    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Reward')
    plt.savefig('trader/training_reward.png')

    plt.clf()
    plt.plot(losses)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('trader/training_loss.png')


def test():
    agent = DQNAgent(state_size=8, action_size=4,
                     epsilon=0.0)
    agent.load(model_save_path)

    env = gym.make('LunarLander-v2', render_mode='human')
    observation, info = env.reset()

    # Test the agent
    for episode in range(10):
        print('Testing Episode:', episode)
        observation, info = env.reset()

        while True:
            action = agent.act(observation)
            observation, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break


# train(num_episodes=300, load_model=False, epsilon=1.0, learning_rate=0.1)
test()
