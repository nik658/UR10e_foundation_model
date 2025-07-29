# example.py
import gymnasium as gym
import gym_xarm
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make("gym_xarm/XarmLift-v0", render_mode="human")
observation, info = env.reset()

model = PPO("MlpPolicy", env, device="cpu", verbose=True)
model.learn(total_timesteps=250_000)


mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)

print("mean sum reward",mean_reward,"std",std_reward)

observation, info = env.reset(seed=42)

for _ in range(10000):
    action, _ = model.predict(observation, deterministic=True)
    observation, reward, terminated, truncated, info = env.step(action)
    image = env.render()

    if terminated or truncated:
        observation, info = env.reset()



env.close()
