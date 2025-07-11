import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.genetic_algorithm import genetic_algorithm

if __name__ == "__main__":
    best = genetic_algorithm(
        pop_size=50,           # pode ajustar para testes rápidos
        individual_size=8,
        n_generations=10,       # pode ajustar para testes rápidos
        mutation_rate=0.1,
        mutation_strength=0.5,
        n_episodes=3           # para rodar mais rápido
    )
    print("Melhor indivíduo final:")
    print("Fitness:", best.fitness)
    print("Genes:", best.genes)