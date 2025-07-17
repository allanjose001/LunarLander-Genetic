import numpy as np
from agents.individual import Individual
from utils.evaluation import evaluate_individual

def read_genes_from_log(filename):
    genes_list = []
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("Genes:"):
                parts = line.strip().split("Genes:")[1]
                genes = np.array([float(x) for x in parts.strip().split()])
                genes_list.append(genes)
    return genes_list

def create_population_from_genes(genes_list, pop_size):
    population = []
    n = len(genes_list)
    for i in range(pop_size):
        genes = genes_list[i % n]  # Repete se necessário
        ind = Individual(genes=genes)
        ind.fitness = evaluate_individual(ind.genes, n_episodes=20)
        population.append(ind)
    return population

if __name__ == "__main__":
    pop_size = 20  # Defina o tamanho desejado da população
    genes_list = read_genes_from_log("best_genes_log.txt")
    population = create_population_from_genes(genes_list, pop_size)
    for i, ind in enumerate(population):
        print(f"Indivíduo {i+1}:")
        print(f"Fitness = {ind.fitness:.2f}")
        print(f"Genes = {ind.genes}\n")