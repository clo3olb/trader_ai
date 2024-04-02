import gymnasium as gym

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy


# Create environment
env = gym.make("LunarLander-v2", render_mode="rgb_array")

# add timeout wrapper
env = gym.wrappers.TimeLimit(
    env, max_episode_steps=10000)

# model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.001)
model = PPO.load("./trader/lunar_project/lunar.pt", env=env)
# model.learn(total_timesteps=1_000_000)
# model.save("./trader/lunar_project/lunar.pt")

mean_reward, std_reward = evaluate_policy(
    model, model.get_env(), n_eval_episodes=10)


# Enjoy trained agent
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(10000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
