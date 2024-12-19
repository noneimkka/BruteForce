from chromosome import Chromosome
from genetic_algorithm import GeneticAlgorithm
import string
import random


def generate_random_genes(length: int) -> str:
    characters = string.ascii_letters + string.digits + string.punctuation
    return "".join(random.choice(characters) for _ in range(length))


def main():
    target_password = input("Введите целевой пароль: ")
    password_length = len(target_password)

    print(f"Длина целевого пароля: {password_length}")

    population_size = 200

    # Инициализация начальной популяции
    initial_population = [
        Chromosome(generate_random_genes(password_length))
        for _ in range(population_size)
    ]

    threshold = password_length  # Максимальный фитнес

    # Определение фитнес-функции
    def fitness(chromosome: Chromosome) -> float:
        return sum(1 for a, b in zip(chromosome.genes, target_password) if a == b)

    ga = GeneticAlgorithm(
        initial_population=initial_population,
        fitness_function=fitness,
        threshold=threshold,
        max_generations=10000,
        mutation_chance=0.1,
        crossover_chance=0.8,
        selection_type=GeneticAlgorithm.SelectionType.TOURNAMENT
    )

    best = ga.run()
    if best.genes == target_password:
        print(f"Найден пароль: {best.genes} с фитнесом: {ga._fitness_function(best)}")
    else:
        print(f"Пароль не найден. Лучший найденный: {best.genes} с фитнесом: {ga._fitness_function(best)}")


if __name__ == "__main__":
    main()
