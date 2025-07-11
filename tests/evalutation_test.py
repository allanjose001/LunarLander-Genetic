import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.evaluation import evaluate_individual

# Cria genes aleatórios para testar
genes = np.random.uniform(-1, 1, 12)

print("Testando avaliação de indivíduo...")
fitness = evaluate_individual(genes, n_episodes=3, render=True)
print("Fitness (média de recompensas):", fitness)
print("Genes testados:", genes)