import gymnasium as gym
from agents.individual import Individual
import numpy as np

def simple_policy(obs, genes):
    scores = obs * genes
    action = np.argmax(scores)
    return int(np.clip(action, 0, 3))

def play_episode(individual, env):
    obs, info = env.reset()
    done = False
    episode_reward = 0
    while not done:
        action = simple_policy(obs, individual.genes)
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        if truncated:
            break
    return episode_reward

env = gym.make("LunarLander-v3", render_mode="human")

# Instancia dois indivíduos
ind1 = Individual(size=8)
ind2 = Individual(size=8)

print("Jogando com ind1...")
reward1 = play_episode(ind1, env)
print("Recompensa ind1:", reward1)
print("Genes ind1:", ind1.genes)

print("\nJogando com ind2...")
reward2 = play_episode(ind2, env)
print("Recompensa ind2:", reward2)
print("Genes ind2:", ind2.genes)

# Cruzamento
child = ind1.crossover(ind2)
print("\nJogando com filho (cruzamento de ind1 e ind2)...")
reward_child = play_episode(child, env)
print("Recompensa filho:", reward_child)
print("Genes filho:", child.genes)

env.close()