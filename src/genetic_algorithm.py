from __future__ import annotations
import logging
from typing import TypeVar, Generic, List, Tuple, Callable
from enum import Enum
from random import choices, random
from heapq import nlargest
from statistics import mean
from chromosome import Chromosome
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import os

C = TypeVar("C", bound=Chromosome)

class GeneticAlgorithm(Generic[C]):
    SelectionType = Enum("SelectionType", "ROULETTE TOURNAMENT")

    def __init__(
        self,
        initial_population: List[C],
        fitness_function: Callable[[C], float],
        threshold: float,
        max_generations: int = 100,
        mutation_chance: float = 0.01,
        crossover_chance: float = 0.7,
        selection_type: SelectionType = SelectionType.TOURNAMENT,
        log_file: str = "logs/genetic_algorithm.log",
    ) -> None:
        self._population: List[C] = initial_population
        self._threshold: float = threshold
        self._max_generations: int = max_generations
        self._mutation_chance: float = mutation_chance
        self._crossover_chance: float = crossover_chance
        self._selection_type: GeneticAlgorithm.SelectionType = selection_type
        self._fitness_function: Callable[[C], float] = fitness_function

        # Создаем директорию logs если она не существует
        os.makedirs("logs", exist_ok=True)

        logging.basicConfig(
            filename=log_file,
            filemode='w',
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO,
            encoding='utf-8'
        )
        logging.info("Начато выполнение генетического алгоритма")

        # Инициализация списков для хранения данных
        self.best_fitness_history: List[float] = []
        self.avg_fitness_history: List[float] = []

    def _pick_roulette(self, wheel: List[float]) -> Tuple[C, C]:
        return tuple(choices(self._population, weights=wheel, k=2))

    def _pick_tournament(self, num_participants: int) -> Tuple[C, C]:
        participants: List[C] = choices(self._population, k=num_participants)
        return tuple(nlargest(2, participants, key=self._fitness_function))

    def _reproduce_and_replace(self) -> None:
        new_population: List[C] = []
        while len(new_population) < len(self._population):
            if self._selection_type == GeneticAlgorithm.SelectionType.ROULETTE:
                parents: Tuple[C, C] = self._pick_roulette(
                    [self._fitness_function(x) for x in self._population]
                )
            else:
                parents = self._pick_tournament(len(self._population) // 2)
            if random() < self._crossover_chance:
                offspring = parents[0].crossover(parents[1])
                new_population.extend(offspring)
            else:
                new_population.extend(parents)
        if len(new_population) > len(self._population):
            new_population.pop()
        self._population = new_population

    def _mutate(self) -> None:
        for individual in self._population:
            if random() < self._mutation_chance:
                individual.mutate()

    def run(self) -> C:
        best: C = max(self._population, key=self._fitness_function)
        start_time = time.time()
        for generation in tqdm(range(self._max_generations), desc="Поколения"):
            if self._fitness_function(best) >= self._threshold:
                logging.info(f"Целевой пароль найден на поколении {generation}")
                break

            # Сохранение данных для визуализации без вывода в консоль
            current_best_fitness = self._fitness_function(best)
            current_avg_fitness = mean(map(self._fitness_function, self._population))
            self.best_fitness_history.append(current_best_fitness)
            self.avg_fitness_history.append(current_avg_fitness)
            
            # Логирование без вывода в консоль
            logging.info(
                f"Поколение {generation}: Лучший фитнес = {current_best_fitness}, "
                f"Средний фитнес = {current_avg_fitness:.2f}"
            )

            # Сохранение лучшей хромосомы
            self._log_best_chromosome(generation, best)

            self._reproduce_and_replace()
            self._mutate()
            highest: C = max(self._population, key=self._fitness_function)
            if self._fitness_function(highest) > self._fitness_function(best):
                best = highest
        else:
            logging.info("Генетический алгоритм завершён без нахождения целевого пароля")

        end_time = time.time()
        total_time = end_time - start_time
        logging.info(f"Общее время выполнения алгоритма: {total_time:.2f} секунд")
        print(f"Общее время выполнения алгоритма: {total_time:.2f} секунд")

        # Визуализация прогресса
        self._plot_progress()

        return best

    def _log_best_chromosome(self, generation: int, best: C) -> None:
        """Записывает информацию о лучшей хромосоме в файл."""
        with open("logs/best_chromosomes.txt", "a", encoding='utf-8') as f:
            f.write(f"Поколение {generation}: {str(best)} с фитнесом {self._fitness_function(best)}\n")

    def _plot_progress(self) -> None:
        plt.figure(figsize=(10, 5))
        plt.plot(self.best_fitness_history, label="Лучший фитнес")
        plt.plot(self.avg_fitness_history, label="Средний фитнес")
        plt.xlabel("Поколение")
        plt.ylabel("Фитнес")
        plt.title("Прогресс Генетического Алгоритма")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("logs/genetic_algorithm_progress.png")  # Сохраняем график
        plt.show()
        logging.info("Визуализация прогресса сохранена как logs/genetic_algorithm_progress.png")