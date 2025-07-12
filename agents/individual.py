import numpy as np

class Individual:
    def __init__(self, genes=None, size=12):
        # Inicializa com vetor de genes aleatórios se não fornecido
        if genes is None:
            self.genes = np.random.uniform(-1, 1, size)
        else:
            self.genes = np.array(genes)
        self.fitness = None

    def clone(self):
        # Retorna uma cópia do indivíduo
        return Individual(genes=self.genes.copy())

    def crossover(self, other):
        # Cruzamento de um ponto
        point = np.random.randint(1, len(self.genes))
        child_genes = np.concatenate((self.genes[:point], other.genes[point:]))
        return Individual(genes=child_genes)

    def evaluate(self, fitness_function):
        # Avalia o fitness usando uma função externa
        self.fitness = fitness_function(self.genes)
        return self.fitness

    def __str__(self):
        return f"Individual(genes={self.genes}, fitness={self.fitness})"