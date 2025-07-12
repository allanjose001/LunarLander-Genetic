import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.genetic_algorithm import genetic_algorithm

if __name__ == "__main__":
    best = genetic_algorithm(
        pop_size=250,
        individual_size=12,
        n_generations=50,
        mutation_rate=0.3,
        mutation_strength=0.7,
        n_episodes=7
    )
    print("Melhor indivíduo final:")
    print("Fitness:", best.fitness)
    print("Genes:", best.genes)