import numpy as np
from agents.individual import Individual

def one_point_crossover(parent1, parent2):
    point = np.random.randint(1, len(parent1.genes))
    child_genes = np.concatenate((parent1.genes[:point], parent2.genes[point:]))
    return Individual(genes=child_genes)

def two_point_crossover(parent1, parent2):
    size = len(parent1.genes)
    p1, p2 = sorted(np.random.choice(range(1, size), 2, replace=False))
    child_genes = parent1.genes.copy()
    child_genes[p1:p2] = parent2.genes[p1:p2]
    return Individual(genes=child_genes)

def uniform_crossover(parent1, parent2, prob=0.5):
    mask = np.random.rand(len(parent1.genes)) < prob
    child_genes = np.where(mask, parent1.genes, parent2.genes)
    return Individual(genes=child_genes)
