import numpy as np
from agents.individual import Individual
from utils.evaluation import evaluate_individual
from agents.crossover import one_point_crossover
from agents.mutation import gaussian_mutation

def read_genes_from_log(filename):
    genes_list = []
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("Genes:"):
                parts = line.strip().split("Genes:")[1]
                genes = np.array([float(x) for x in parts.strip().split()])
                genes_list.append(genes)
    return genes_list

def create_population_from_genes(genes_list, pop_size, n_episodes=3):
    population = []
    n = len(genes_list)
    for i in range(pop_size):
        genes = genes_list[i % n]  # Repete se necessário
        ind = Individual(genes=genes)
        ind.fitness = evaluate_individual(ind.genes, n_episodes=n_episodes)
        population.append(ind)
    return population

if __name__ == "__main__":
    pop_size = 50
    n_episodes = 15
    n_generations = 20
    mutation_rate = 0.3
    mutation_strength = 0.3

    genes_list = read_genes_from_log("best_genes_log.txt")
    population = create_population_from_genes(genes_list, pop_size, n_episodes=n_episodes)

    for gen in range(n_generations):
        # Mantém apenas indivíduos com fitness >= 220
        population = [ind for ind in population if ind.fitness >= 220]

        # Se não houver bons, pare
        if not population:
            print(f"Nenhum indivíduo com fitness >= 220 na geração {gen+1}!")
            break

        # Preenche a população com filhos de crossover + mutação
        while len(population) < pop_size:
            parent1, parent2 = np.random.choice(population, 2, replace=True)
            child = one_point_crossover(parent1, parent2)
            # Aplica mutação gaussiana
            child.genes = gaussian_mutation(
                child.genes,
                mutation_rate=mutation_rate,
                mutation_strength=mutation_strength
            )
            child.fitness = evaluate_individual(child.genes, n_episodes=n_episodes)
            population.append(child)

        # (Opcional) Mostra progresso
        best = max(population, key=lambda ind: ind.fitness)
        print(f"Geração {gen+1}: Melhor fitness = {best.fitness:.2f}")

    # Ordena e imprime os 10 melhores ao final
    population.sort(key=lambda ind: ind.fitness, reverse=True)
    print("\nTop 10 indivíduos da população final:")
    for i, ind in enumerate(population[:10]):
        print(f"Indivíduo {i+1}:")
        print(f"Fitness = {ind.fitness:.2f}")
        print(f"Genes = {ind.genes}\n")

    # Salva o melhor indivíduo no arquivo de log
    best = population[0]
    with open("best_genes_log.txt", "a") as f:
        f.write(f"Fitness: {best.fitness:.2f}\n")
        f.write("Genes: " + " ".join([f"{g:.6f}" for g in best.genes]) + "\n")
        f.write("-" * 40 + "\n")