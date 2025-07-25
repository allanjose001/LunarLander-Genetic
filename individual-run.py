import numpy as np
import gymnasium as gym
from agents.policy import combined_policy

def render_individual(genes, n_episodes=1):
    env = gym.make("LunarLander-v3", render_mode="human")
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = combined_policy(obs, genes)
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            if truncated:
                break
        print(f"Episódio {ep+1} - Recompensa: {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    genes = [-0.30355334, -0.97991491, -0.01424614, -1.70135548, -0.10998174,  0.58534388,
              0.07423689,  1.46115483,  0.81766629, -0.35724814,  0.28569351,  0.62564056,
              0.15626767,  0.36319294, -0.06410646, -0.52669254,  1.32678994, -0.55979689,
              -0.58644915]
    render_individual(np.array(genes), n_episodes=10)