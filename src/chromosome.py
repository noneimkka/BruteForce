import string
import random


class Chromosome:
    def __init__(self, genes: str):
        self.genes = genes

    def mutate(self, mutation_rate: float = 0.01) -> None:
        genes = list(self.genes)
        for i in range(len(genes)):
            if random.random() < mutation_rate:
                genes[i] = random.choice(
                    string.ascii_letters + string.digits + string.punctuation
                )
        self.genes = "".join(genes)

    def crossover(self, other: "Chromosome") -> tuple["Chromosome", "Chromosome"]:
        if len(self.genes) != len(other.genes):
            raise ValueError("Длины генов родителей должны совпадать.")
        crossover_point = random.randint(1, len(self.genes) - 1)
        child1_genes = self.genes[:crossover_point] + other.genes[crossover_point:]
        child2_genes = other.genes[:crossover_point] + self.genes[crossover_point:]
        return Chromosome(child1_genes), Chromosome(child2_genes)

    def __str__(self):
        return self.genes
