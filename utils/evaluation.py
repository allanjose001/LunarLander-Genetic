import gymnasium as gym
import numpy as np
from agents.policy import combined_policy  # Importa a nova política

def evaluate_individual(genes, n_episodes=3, render=False):
    env = gym.make("LunarLander-v3", render_mode="human" if render else None)
    total_reward = 0
    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = combined_policy(obs, genes)  # Usa a nova política
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            if truncated:
                break
        """
        # Recompensa por tocar o solo com as duas pernas
        if obs[6] == 1 and obs[7] == 1:
            reward_legs = 100  # valor ajustável
            episode_reward += reward_legs

        # Recompensa por pousar próximo ao centro
        reward_center = 100 * (1 - abs(obs[0]))  # quanto mais perto do centro, maior a recompensa
        episode_reward += reward_center
        """
        # Recompensa por pousar suavemente (baixa velocidade vertical)
        reward_soft = 100 * (1 - abs(obs[3]))  # quanto menor a velocidade vertical, maior a recompensa
        episode_reward += reward_soft
        
        total_reward += episode_reward / n_episodes
    env.close()
    # Retorna a média das recompensas como fitness
    return total_reward