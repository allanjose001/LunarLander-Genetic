from agents.genetic_algorithm import genetic_algorithm
from agents.selection import tournament_selection, roulette_selection
from agents.crossover import one_point_crossover, two_point_crossover, uniform_crossover
from agents.mutation import gaussian_mutation, random_reset_mutation, non_uniform_mutation

def get_float(prompt, default):
    try:
        val = input(f"{prompt} (padrão = {default}): ")
        return float(val) if val.strip() else default
    except ValueError:
        print("Entrada inválida. Usando valor padrão.")
        return default

def get_int(prompt, default):
    try:
        val = input(f"{prompt} (padrão = {default}): ")
        return int(val) if val.strip() else default
    except ValueError:
        print("Entrada inválida. Usando valor padrão.")
        return default

def select_option(prompt, options):
    print(prompt)
    for i, (name, _) in enumerate(options):
        print(f"{i + 1}. {name}")
    idx = int(input("Escolha uma opção: ")) - 1
    return options[idx][1]

if __name__ == "__main__":
    selection_methods = [
        ("Torneio", tournament_selection),
        ("Roleta", roulette_selection),
    ]

    crossover_methods = [
        ("Crossover de 1 ponto", one_point_crossover),
        ("Crossover de 2 pontos", two_point_crossover),
        ("Crossover uniforme", lambda p1, p2: uniform_crossover(p1, p2, prob=0.5)),
    ]

    mutation_methods = [
        ("Mutação Gaussiana", gaussian_mutation),
        ("Reset Aleatório", random_reset_mutation),
        ("Mutação Não-Uniforme", non_uniform_mutation),
    ]

    selection = select_option("Selecione o método de seleção:", selection_methods)
    crossover = select_option("Selecione o método de crossover:", crossover_methods)
    mutation = select_option("Selecione o método de mutação:", mutation_methods)

    pop_size = get_int("Tamanho da população", 50)
    n_generations = get_int("Número de gerações", 25)
    mutation_rate = get_float("Taxa de mutação (0.0 a 1.0)", 0.7)
    mutation_strength = get_float("Força da mutação", 0.5)
    n_episodes = get_int("Número de episódios por avaliação", 7)

    # kwargs para mutação
    mutation_kwargs = {}
    if mutation == non_uniform_mutation:
        mutation_kwargs = {
            "generation": 0,
            "max_generations": n_generations
        }

    best = genetic_algorithm(
        pop_size=pop_size,
        individual_size=18,
        n_generations=n_generations,
        mutation_rate=mutation_rate,
        mutation_strength=mutation_strength,
        n_episodes=n_episodes,
        selection_func=selection,
        crossover_func=crossover,
        mutation_func=mutation,
        selection_kwargs={"tournament_size": 7} if selection == tournament_selection else {},
        mutation_kwargs=mutation_kwargs
    )

    print("Melhor indivíduo final:")
    print("Fitness:", best.fitness)
    print("Genes:", best.genes)