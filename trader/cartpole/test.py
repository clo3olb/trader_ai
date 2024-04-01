import gymnasium as gym

from stable_baselines3 import DQN, PPO

env = gym.make("CartPole-v1", render_mode="human")

model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.01)
# model.load("./trader/cartpole/dqn_model.pt")
model.learn(total_timesteps=25_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(10000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")

    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()

model.save("./trader/cartpole/dqn_model.pt")
