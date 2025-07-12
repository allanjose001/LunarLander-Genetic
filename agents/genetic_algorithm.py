from agents.population import Population
from agents.selection import roulette_selection
from utils.evaluation import evaluate_individual
from agents.selection import tournament_selection
from agents.mutation import *

def genetic_algorithm(
    pop_size=50,
    individual_size=12,
    n_generations=20,
    mutation_rate=0.1,
    mutation_strength=0.5,
    n_episodes=3
):
    # Inicializa população
    population = Population(size=pop_size, individual_size=individual_size)

    # Avalia fitness inicial
    for ind in population.individuals:
        ind.fitness = evaluate_individual(ind.genes, n_episodes=n_episodes)

    for gen in range(n_generations):
        new_population = Population(size=0, individual_size=individual_size)

        # ELITISMO: copia os melhores 5% para a nova população
        n_elite = max(1, int(0.01 * pop_size))
        population.sort_by_fitness(reverse=True)
        for ind in population.individuals[:n_elite]:
            elite = ind.clone()
            elite.fitness = ind.fitness
            new_population.add(elite)

        # Preenche o restante da população normalmente
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, tournament_size=7)
            parent2 = tournament_selection(population, tournament_size=7)
            #parent1 = roulette_selection(population)
            #parent2 = roulette_selection(population)

            child = parent1.crossover(parent2)
            child.genes = gaussian_mutation(child.genes, mutation_rate=mutation_rate, mutation_strength=mutation_strength)
            #child.genes = mutation.random_reset_mutation(child.genes, mutation_rate=0.1, bounds=bounds)
            #child.genes = non_uniform_mutation(child.genes, mutation_rate=mutation_rate, mutation_strength=mutation_strength, generation=gen, max_generations=n_generations)
            child.fitness = evaluate_individual(child.genes, n_episodes=n_episodes)
            new_population.add(child)

        population = new_population

        # Imprime o melhor indivíduo da geração
        best = population.best_individual()
        print(f"Geração {gen+1}: Melhor fitness = {best.fitness:.2f}")
        print(f"Genes: {best.genes}\n")
        # Removido o render de cada geração

    
    while True:
        user_input = input("Digite 1 para renderizar o melhor agente ou 0 para sair: ")
        if user_input == "1":
            evaluate_individual(population.best_individual().genes, n_episodes=1, render=True)
        elif user_input == "0":
            print("Encerrando o programa.")
            break
        else:
            print("Entrada inválida. Digite 1 ou 0.")
    best_final = population.best_individual()
    #print("Pronto para renderizar o melhor indivíduo final!")
    #input("Pressione ENTER para visualizar o melhor agente...")
    #evaluate_individual(best_final.genes, n_episodes=5, render=True)
    return best_final