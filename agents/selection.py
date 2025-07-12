import numpy as np

def roulette_selection(population):
    fitness_values = np.array(population.get_all_fitness())
    # Ajusta fitness para evitar valores negativos
    min_fitness = np.min(fitness_values)
    if min_fitness < 0:
        fitness_values = fitness_values - min_fitness + 1e-6
    total_fitness = np.sum(fitness_values)
    if total_fitness == 0:
        # Se todos os fitness são zero, seleciona aleatoriamente
        idx = np.random.randint(len(population))
        return population[idx]
    probabilities = fitness_values / total_fitness
    idx = np.random.choice(len(population), p=probabilities)
    return population[idx]


def tournament_selection(population, tournament_size=3):
    # Escolhe indivíduos aleatórios
    selected_indices = np.random.choice(len(population), size=tournament_size, replace=False)
    selected_individuals = [population[idx] for idx in selected_indices]
    # Seleciona o melhor do torneio
    best_individual = max(selected_individuals, key=lambda ind: ind.fitness)
    return best_individual