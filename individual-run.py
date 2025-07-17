import numpy as np
import gymnasium as gym
from agents.policy import combined_policy

def render_individual(genes, n_episodes=1):
    env = gym.make("LunarLander-v3", render_mode="human")
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = combined_policy(obs, genes)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            if truncated:
                break
        print(f"Episódio {ep+1} - Recompensa: {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    # Substitua pelos genes desejados (exemplo abaixo)
    genes = [
        0.178824, 0.320984, -0.694198, -0.949093, 0.251090, 0.011668, -0.532903, -0.534955,
        -0.431704, 1.058738, 1.065251, -0.800082, -0.438952, 0.696824, 0.697558, 0.444008,
        0.101568, 0.034853, 0.652524
    ]
    render_individual(np.array(genes), n_episodes=1)