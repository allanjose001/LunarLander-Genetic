from agents.individual import Individual

class Population:
    def __init__(self, size=50, individual_size=12):
        self.size = size
        self.individual_size = individual_size
        self.individuals = [Individual(size=individual_size) for _ in range(size)]

    def __len__(self):
        return len(self.individuals)

    def __getitem__(self, idx):
        return self.individuals[idx]

    def get_all_genes(self):
        return [ind.genes for ind in self.individuals]

    def get_all_fitness(self):
        return [ind.fitness for ind in self.individuals]

    def add(self, individual):
        self.individuals.append(individual)

    def clear(self):
        self.individuals = []

    def best_individual(self):
        return max(self.individuals, key=lambda ind: ind.fitness if ind.fitness is not None else float('-inf'))

    def sort_by_fitness(self, reverse=True):
        self.individuals.sort(key=lambda ind: ind.fitness if ind.fitness is not None else float('-inf'), reverse=reverse)