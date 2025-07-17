import numpy as np

def gaussian_mutation(genes, mutation_rate=0.1, mutation_strength=0.5):
    new_genes = genes.copy()
    for i in range(len(new_genes)):
        if np.random.rand() < mutation_rate:
            new_genes[i] += np.random.normal(0, mutation_strength)
    return new_genes

# Mutação por reset aleatorio (valor novo dentro de bounds)
def random_reset_mutation(genes, mutation_rate=0.1, mutation_strength=0.5, bounds=None):
    if bounds is None:
        # Usar bounds padrão baseado nos valores atuais dos genes
        bounds = [(-2.0, 2.0)] * len(genes)
    
    new_genes = genes.copy()
    for i in range(len(new_genes)):
        if np.random.rand() < mutation_rate:
            new_genes[i] = np.random.uniform(bounds[i][0], bounds[i][1])
    return new_genes

# Mutação não-uniforme (exploração no início, refinamento no fim)
def non_uniform_mutation(genes, mutation_rate=0.1, mutation_strength=0.5, generation=1, max_generations=100):
    new_genes = genes.copy()
    b = 3  # controle de intensidade
    for i in range(len(new_genes)):
        if np.random.rand() < mutation_rate:
            delta = np.random.uniform(-1, 1) * mutation_strength
            factor = (1 - generation / max_generations) ** b
            new_genes[i] += delta * factor
    return new_genes