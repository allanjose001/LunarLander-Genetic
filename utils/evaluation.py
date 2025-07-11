import gymnasium as gym
import numpy as np

def simple_policy(obs, genes):
    scores = obs * genes
    action = np.argmax(scores)
    return int(np.clip(action, 0, 3))

def evaluate_individual(genes, n_episodes=3, render=False):
    env = gym.make("LunarLander-v3", render_mode="human" if render else None)
    total_reward = 0
    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = simple_policy(obs, genes)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            if truncated:
                break
        total_reward += episode_reward/n_episodes
    env.close()
    # Retorna a média das recompensas como fitness
    return total_reward