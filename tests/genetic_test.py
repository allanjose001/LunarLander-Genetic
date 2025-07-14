import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.genetic_algorithm import genetic_algorithm

if __name__ == "__main__":
    best = genetic_algorithm(
        pop_size=120,
        individual_size=15,
        n_generations=25,
        mutation_rate=0.7,
        mutation_strength=0.1,
        n_episodes=10
    )
    print("Melhor indivíduo final:")
    print("Fitness:", best.fitness)
    print("Genes:", best.genes)