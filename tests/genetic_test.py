import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.genetic_algorithm import genetic_algorithm

if __name__ == "__main__":
    best = genetic_algorithm(
        pop_size=50,
        individual_size=12,
        n_generations=10,
        mutation_rate=0.1,

        mutation_strength=0.5,
        n_episodes=5
    )
    print("Melhor indivíduo final:")
    print("Fitness:", best.fitness)
    print("Genes:", best.genes)