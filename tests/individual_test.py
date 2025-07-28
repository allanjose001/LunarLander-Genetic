import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
from agents.individual import Individual
from agents.policy import *
import numpy as np

def play_episode(individual, env):
    obs, info = env.reset()
    done = False
    episode_reward = 0
    while not done:
        action = combined_policy(obs, individual.genes)
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        if truncated:
            break
    return episode_reward

# COLE AQUI OS GENES DO INDIVÍDUO TREINADO
trained_genes = [0.178824, 0.320984, -0.694198, -0.949093, 0.251090, 
    0.011668, -0.532903, -0.534955, -0.431704, 1.058738, 
    1.065251, -0.800082, -0.438952, 0.696824, 0.697558, 
    0.444008, 0.101568, 0.034853, 0.652524]
# Número de execuções
n_executions = 4

env = gym.make("LunarLander-v3", render_mode="human")

# Cria o indivíduo com os genes treinados
trained_individual = Individual(genes=trained_genes)

print("=== TESTANDO INDIVÍDUO TREINADO ===")
print(f"Genes: {trained_individual.genes}")
print(f"Realizando {n_executions} execuções...\n")

rewards = []

for i in range(n_executions):
    print(f"Execução {i+1}...")
    reward = play_episode(trained_individual, env)
    rewards.append(reward)
    print(f"Recompensa: {reward:.2f}")

print(f"\n=== RESULTADOS ===")
print(f"Recompensas: {[f'{r:.2f}' for r in rewards]}")
print(f"Recompensa máxima: {np.max(rewards):.2f}")
print(f"Desvio padrão: {np.std(rewards):.2f}")

env.close()