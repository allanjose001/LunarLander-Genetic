import gymnasium as gym
from agents.policy import combined_policy

def evaluate_individual(genes, n_episodes=3, render=False):
    env = gym.make("LunarLander-v3", render_mode="human" if render else None)
    total_reward = 0
    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = combined_policy(obs, genes, include_aux_reward=True)
            obs, reward, done, truncated, info = env.step(action)
            aux_reward = 0.0
            episode_reward += reward
            if truncated:
                break
        
        total_reward += episode_reward / n_episodes

    env.close()
    # Retorna a média das recompensas como fitness
    return total_reward