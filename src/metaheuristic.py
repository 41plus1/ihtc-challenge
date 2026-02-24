import numpy as np
import pandas as pd
from typing import List

from src.data_loader import InstanceData
from src.experiments import evaluate_solution

class GeneticAlgorithmNRA:
    def __init__(self, inst: InstanceData, seed: int, pop_size: int, generations: int, mutation_rate: float):
        self.inst = inst
        self.rng = np.random.default_rng(seed)
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate

        # mapeamento das tarefas
        self.tasks = self.inst.occupied_room_shifts[["global_shift", "room_id"]].to_dict("records")

        # mapeia enfermeiros disponíveis por turno para garantir a restrição de disponibilidade
        self.available_nurses = (
            self.inst.nurse_shifts.groupby("global_shift")["nurse_id"]
            .apply(list)
            .to_dict()
        )

    def generate_individual(self) -> List[str]:
        """gera um cromossomo válido: uma lista de nurse_ids paralela à lista de tarefas"""
        individual = []

        for task in self.tasks:
            shift = task["global_shift"]
            available = self.available_nurses[shift]
            individual.append(self.rng.choice(available))

        return individual
        
    def evaluate(self, individual: List[str]) -> float:
        """avalia o indivíduo usando a função objetivo"""
        df_sol = pd.DataFrame(self.tasks)
        df_sol["nurse_id"] = individual
        breakdown = evaluate_solution(self.inst, df_sol)

        return  breakdown.total
        
    def crossover(self, parent1: List[str], parent2: List[str]) -> List[str]:
        """crossover Uniforme: escolhe aleatoriamente o gene de um dos pais"""
        child = []

        for g1, g2 in zip(parent1, parent2):
            child.append(g1 if self.rng.random() < 0.5 else g2)

        return child
        
    def mutate(self, individual: List[str]) -> List[str]:
        """mutação: altera o enfermeiro de uma tarefa por outro disponível no mesmo turno"""
        mutated = []

        for i, task in enumerate(self.tasks):
            if self.rng.random() < self.mutation_rate:
                # troca por outro enfermeiro válido do mesmo turno
                shift = task["global_shift"]
                available = self.available_nurses[shift]
                mutated.append(self.rng.choice(available))
            else:
                mutated.append(individual[i])

        return mutated
        
    def solve(self) -> pd.DataFrame:
        """executa o ciclo evolutivo e retorna a melhor solução encontrada"""
        # inicializa a população e calcula os custos
        population = [self.generate_individual() for _ in range(self.pop_size)]
        fitnesses = [self.evaluate(ind) for ind in population]

        best_idx = np.argmin(fitnesses)
        best_individual = population[best_idx]
        best_fitness = fitnesses[best_idx]

        # ciclo de evolução
        for _ in range(self.generations):
            new_population = []

            # elitismo: mantém a melhor solução da geração anterior
            new_population.append(best_individual)

            while len(new_population) < self.pop_size:
                # torneio de tamanho 2 para selecionar os pais
                i1, i2 = self.rng.choice(self.pop_size, 2, replace = False)
                p1 = population[i1] if fitnesses[i1] < fitnesses[i2] else population[i2]

                i3, i4 = self.rng.choice(self.pop_size, 2, replace = False)
                p2 = population[i3] if fitnesses[i3] < fitnesses[i4] else population[i4]

                # cruzamento e mutação
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                new_population.append(child)

            population = new_population
            fitnesses = [self.evaluate(ind) for ind in population]

            # atualização do melhor global
            current_best_idx = np.argmin(fitnesses)

            if fitnesses[current_best_idx] < best_fitness:
                best_fitness = fitnesses[current_best_idx]
                best_individual = population[current_best_idx]
            
        df_best = pd.DataFrame(self.tasks)
        df_best["nurse_id"] = best_individual

        return df_best
        
def ga_solver(inst: InstanceData, seed: int) -> pd.DataFrame:
    """função de interface para injeção no pipeline de experimentos"""
    # paramêtros ajustáveis do algoritmo genético
    ga = GeneticAlgorithmNRA(
        inst = inst,
        seed = seed,
        pop_size = 40,          # tamanho da população
        generations = 50,       # número de iterações
        mutation_rate = 0.05    # taxa de mutação
    )

    return ga.solve()