from agents.population import Population, Individual
from agents.selection import *
from utils.evaluation import evaluate_individual
from agents.selection import tournament_selection
from agents.mutation import *
from agents.crossover import *
import time
import numpy as np
import matplotlib.pyplot as plt

def genetic_algorithm(
    pop_size=50,
    individual_size=19,
    n_generations=20,
    mutation_rate=0.1,
    mutation_strength=0.5,
    n_episodes=3,
    selection_func=tournament_selection,
    crossover_func=uniform_crossover,
    mutation_func=gaussian_mutation,
    selection_kwargs=None,
    crossover_kwargs=None,
    mutation_kwargs=None,
):
    selection_kwargs = selection_kwargs or {}
    crossover_kwargs = crossover_kwargs or {}
    mutation_kwargs = mutation_kwargs or {}

    # Contador para saber a duração do treinamento
    start_time = time.time()

    # Inicializa a população
    population = Population(size=pop_size, individual_size=individual_size)

    # best_fitness = None
    # stagnation_counter = 0

    # Variável para armazenar o melhor indivíduo global
    best_global = None

    # Listas para armazenar os dados de convergência
    best_fitness_history = []
    mean_fitness_history = []

    # Avalia o fitness inicial
    for ind in population.individuals:
        ind.fitness = evaluate_individual(ind.genes, n_episodes=n_episodes)
        if best_global is None or ind.fitness > best_global.fitness:
            best_global = ind.clone()
            best_global.fitness = ind.fitness

    for gen in range(n_generations):
        new_population = Population(size=0, individual_size=individual_size)

        # Aplica elitismo copiando os melhores 1% da população para a nova geração
        apply_elitism(population, pop_size, new_population)

        # Aplica um refinamento leve nos melhores indivíduos da população
        apply_refinement(population, new_population, n_episodes)

        # Preenche o restante da população usando a função auxiliar
        best_global = fill_population(
            population, new_population, pop_size,
            selection_func, crossover_func, mutation_func,
            mutation_rate, mutation_strength, mutation_kwargs, crossover_kwargs, selection_kwargs,
            n_episodes, best_global
        )

        population = new_population

        best = population.best_individual()
        best_fitness_history.append(best.fitness)
        mean_fitness = np.mean([ind.fitness for ind in population.individuals])
        mean_fitness_history.append(mean_fitness)

        # Atualiza o melhor global se necessário (deve ficar aqui!)
        if best_global is None or best.fitness > best_global.fitness:
            best_global = best.clone()
            best_global.fitness = best.fitness

        print(f"Geração {gen+1}: Melhor fitness = {best.fitness:.2f}")
        print(f"Genes: {best.genes}\n")

        # Controle de estagnação
        # population, best_fitness, stagnation_counter = handle_stagnation(
        #     population,
        #     best_fitness,
        #     best.fitness,
        #     stagnation_counter,
        #     pop_size, n_elite,
        #     individual_size,
        #     n_episodes
        # )

    total_time = time.time() - start_time
    total_minutes = int(total_time // 60)
    total_seconds = total_time % 60

    print(f"=== TREINAMENTO CONCLUÍDO ===")
    print(f"Tempo total de treinamento: {total_minutes}m {total_seconds:.1f}s")
    print(f"Melhor fitness final da última geração: {population.best_individual().fitness:.2f}")
    print(f"Melhor fitness global: {best_global.fitness:.2f}")
    print("="*30)

    # Plota a curva de convergência
    plt.figure(figsize=(10, 6))
    plt.plot(best_fitness_history, label="Melhor fitness")
    plt.plot(mean_fitness_history, label="Fitness médio")
    plt.xlabel("Geração")
    plt.ylabel("Fitness")
    plt.title("Curva de Convergência do Algoritmo Genético")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("Melhor indivíduo global de todas as gerações:")
    print("Fitness:", best_global.fitness)
    print("Genes:", best_global.genes)

    while True:
        user_input = input("Digite 1 para renderizar o melhor agente global ou 0 para sair: ")
        if user_input == "1":
            evaluate_individual(best_global.genes, n_episodes=1, render=True)
        elif user_input == "0":
            print("Encerrando o programa.")
            break
        else:
            print("Entrada inválida. Digite 1 ou 0.")
    return best_global

# Função auxiliar para preencher o restante da população
def fill_population(population, new_population, pop_size, selection_func, crossover_func, mutation_func,
                   mutation_rate, mutation_strength, mutation_kwargs, crossover_kwargs, selection_kwargs,
                   n_episodes, best_global):
    """
    Preenche o restante da população com filhos gerados por seleção, crossover e mutação.
    Atualiza best_global se encontrar um indivíduo melhor.
    Retorna o best_global atualizado.
    """
    while len(new_population) < pop_size:
        parent1 = selection_func(population, **selection_kwargs)
        parent2 = selection_func(population, **selection_kwargs)

        child = crossover_func(parent1, parent2, **crossover_kwargs)
        child.genes = mutation_func(
            child.genes,
            mutation_rate=mutation_rate,
            mutation_strength=mutation_strength,
            **mutation_kwargs
        )
        child.fitness = evaluate_individual(child.genes, n_episodes=n_episodes)
        new_population.add(child)

        # Atualiza o melhor global se necessário
        if best_global is None or child.fitness > best_global.fitness:
            best_global = child.clone()
            best_global.fitness = child.fitness
    return best_global

# Aplica elitismo copiando os melhores 1% da população para a nova geração
def apply_elitism(population, pop_size, new_population):
    n_elite = max(1, int(0.01 * pop_size))
    population.sort_by_fitness(reverse=True)
    for ind in population.individuals[:n_elite]:
        elite = ind.clone()
        elite.fitness = ind.fitness
        new_population.add(elite)
    return n_elite

# Aplica um refinamento leve nos melhores indivíduos da população
def apply_refinement(population, new_population, n_episodes, n_refine=2, mutation_strength_refine=0.04):
    for ind in population.individuals[:n_refine]:
        refined_genes = gaussian_mutation(ind.genes, mutation_rate=0.2, mutation_strength=mutation_strength_refine)
        refined = Individual(genes=refined_genes)
        refined.fitness = evaluate_individual(refined.genes, n_episodes=n_episodes)
        new_population.add(refined)

# Lida com estagnação caso 4 gerações consecutivas não melhorem o fitness
def handle_stagnation(population, best_fitness, current_best_fitness, stagnation_counter, pop_size, n_elite, individual_size, n_episodes):
    if best_fitness is None or current_best_fitness > best_fitness:
        best_fitness = current_best_fitness
        stagnation_counter = 0
    else:
        stagnation_counter += 1

    if stagnation_counter >= 4:
        print("Estagnação detectada! Resetando 50% da população...")
        n_reset = int(0.5 * pop_size)

        # Mantém os elitistas
        population.sort_by_fitness(reverse=True)
        elitists = population.individuals[:n_elite]

        # Gera novos indivíduos aleatórios
        new_individuals = [Individual(size=individual_size) for _ in range(n_reset)]

        # Preenche o restante com os melhores não elitistas
        remaining = population.individuals[n_elite:pop_size-n_reset]

        # Nova população
        population.individuals = elitists + remaining + new_individuals

        # Avalia o fitness dos novos indivíduos
        for ind in new_individuals:
            ind.fitness = evaluate_individual(ind.genes, n_episodes=n_episodes)

        stagnation_counter = 0

    return population, best_fitness, stagnation_counter