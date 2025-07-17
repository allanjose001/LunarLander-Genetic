from agents.population import Population, Individual
from agents.selection import *
from utils.evaluation import evaluate_individual
from agents.selection import tournament_selection
from agents.mutation import *
from agents.crossover import *
import time

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

    # Contador pra saber a duração do treinamento
    start_time = time.time()

    # Inicializa população
    population = Population(size=pop_size, individual_size=individual_size)

    best_fitness = None
    stagnation_counter = 0

    # Avalia fitness inicial
    for ind in population.individuals:
        ind.fitness = evaluate_individual(ind.genes, n_episodes=n_episodes)

    for gen in range(n_generations):
        new_population = Population(size=0, individual_size=individual_size)

        n_elite = apply_elitism(population, pop_size, new_population)

        apply_refinement(population, new_population, n_episodes)

        # Preenche o restante da população normalmente
        while len(new_population) < pop_size:
            parent1 = selection_func(population, **selection_kwargs)
            parent2 = selection_func(population, **selection_kwargs)

            child = crossover_func(parent1, parent2, **crossover_kwargs)
            child.genes = mutation_func(child.genes, 
                mutation_rate=mutation_rate, 
                mutation_strength=mutation_strength,
                **mutation_kwargs
            )
 
            child.fitness = evaluate_individual(child.genes, n_episodes=n_episodes)
            new_population.add(child)

        population = new_population

        best = population.best_individual()
        print(f"Geração {gen+1}: Melhor fitness = {best.fitness:.2f}")
        print(f"Genes: {best.genes}\n")

        # Controle de estagnação
        population, best_fitness, stagnation_counter = handle_stagnation(
            population,
            best_fitness, 
            best.fitness, 
            stagnation_counter,
            pop_size, n_elite, 
            individual_size, 
            n_episodes
        )

        total_time = time.time() - start_time
    total_minutes = int(total_time // 60)
    total_seconds = total_time % 60
    
    print(f"=== TREINAMENTO CONCLUÍDO ===")
    print(f"Tempo total de treinamento: {total_minutes}m {total_seconds:.1f}s")
    print(f"Melhor fitness final: {population.best_individual().fitness:.2f}")
    print("="*30)

    while True:
        user_input = input("Digite 1 para renderizar o melhor agente ou 0 para sair: ")
        if user_input == "1":
            evaluate_individual(population.best_individual().genes, n_episodes=1, render=True)
        elif user_input == "0":
            print("Encerrando o programa.")
            break
        else:
            print("Entrada inválida. Digite 1 ou 0.")
    return population.best_individual()

#Aplica elitismo copiando os melhores 5% da população para a nova geração.
def apply_elitism(population, pop_size, new_population):
    n_elite = max(1, int(0.05 * pop_size))
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

# Lida com estagnação caso 4 gerações consecutivas não melhorem de fitness
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
        elitistas = population.individuals[:n_elite]
        
        # Gera novos indivíduos aleatórios
        novos = [Individual(size=individual_size) for _ in range(n_reset)]
        
        # Preenche o restante com os melhores não elitistas
        restantes = population.individuals[n_elite:pop_size-n_reset]
        
        # Nova população
        population.individuals = elitistas + restantes + novos
        
        # Avalia fitness dos novos indivíduos
        for ind in novos:
            ind.fitness = evaluate_individual(ind.genes, n_episodes=n_episodes)
        
        stagnation_counter = 0
    
    return population, best_fitness, stagnation_counter
