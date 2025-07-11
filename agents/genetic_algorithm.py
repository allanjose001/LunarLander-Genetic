from agents.population import Population
from agents.selection import roulette_selection
from utils.evaluation import evaluate_individual

def genetic_algorithm(
    pop_size=50,
    individual_size=8,
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
        while len(new_population) < pop_size:
            # Seleção dos pais
            parent1 = roulette_selection(population)
            parent2 = roulette_selection(population)
            # Reprodução
            child = parent1.crossover(parent2)
            # Mutação
            child.mutate(mutation_rate=mutation_rate, mutation_strength=mutation_strength)
            # Avaliação
            child.fitness = evaluate_individual(child.genes, n_episodes=n_episodes)
            # Adiciona filho à nova população
            new_population.add(child)

        population = new_population

        # Imprime o melhor indivíduo da geração
        best = population.best_individual()
        print(f"Geração {gen+1}: Melhor fitness = {best.fitness:.2f}")
        print(f"Genes: {best.genes}\n")
        evaluate_individual(best.genes, n_episodes=1, render=True)

    # Retorna o melhor indivíduo final
    return population.best_individual()