import os
import json
import numpy as np
import pandas as pd
import pygad
from pathlib import Path

from src.data_loader import InstanceData

class GeneticAlgorithmOptimizer:
    """
    Otimizador baseado em Algoritmo Genético para o problema de alocação de enfermeiros (NRA).
    
    Attributes:
        inst (InstanceData): Os dados da instância do problema.
        seed (int): Semente para reprodutibilidade estocástica.
        pop_size (int): Tamanho da população (número de cromossomas).
        generations (int): Número de gerações para o critério de paragem.
        mutation_rate (float): Probabilidade de mutação de cada gene.
    """
    def __init__(self, inst: InstanceData, seed: int, pop_size: int = 50, generations: int = 100, mutation_rate: float = 0.15):
        self.inst = inst
        self.seed = seed
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        
        # extração das tarefas (quartos a cobrir)
        df_tasks = self.inst.occupied_room_shifts[["global_shift", "room_id", "total_room_workload", "max_skill_required"]]
        self.tasks_df = df_tasks
        self.shifts = df_tasks["global_shift"].to_numpy()
        self.rooms = df_tasks["room_id"].to_numpy()
        self.workloads = df_tasks["total_room_workload"].to_numpy()
        self.req_skills = df_tasks["max_skill_required"].to_numpy()
        self.num_tasks = len(self.shifts)

        # mapeamento de identificadores 
        self.unique_nurses = self.inst.nurse_shifts["nurse_id"].unique()
        self.nurse_id_to_int = {nid: i for i, nid in enumerate(self.unique_nurses)}
        self.nurse_int_to_id = {i: nid for i, nid in enumerate(self.unique_nurses)}

        # pré-computação para cálculo de fitness
        self.nurse_skills = {}
        self.nurse_max_loads = {}
        for _, row in self.inst.nurse_shifts.iterrows():
            shift = row["global_shift"]
            n_int = self.nurse_id_to_int[row["nurse_id"]]
            key = (shift, n_int)
            self.nurse_skills[key] = row["skill_level"]
            self.nurse_max_loads[key] = row["max_load"]
            
        w = self.inst.info.get("weights", {})
        self.w_skill = float(w.get("S2_room_nurse_skill", 1.0))
        self.w_work = float(w.get("S4_nurse_excessive_workload", 1.0))

        # construção do Gene Space (garantia de viabilidade das soluções)
        available_nurses_int = {}
        for shift, group in self.inst.nurse_shifts.groupby("global_shift"):
            available_nurses_int[shift] = [self.nurse_id_to_int[n] for n in group["nurse_id"]]

        self.gene_space = [available_nurses_int[shift] for shift in self.shifts]

    def _fitness_function(self, ga_instance, solution, solution_idx):
        """Avalia a solução gerada. Retorna o custo negativo (PyGAD maximiza)."""
        skill_deficit = 0.0
        nurse_workloads = {}
        
        for i, nurse_int in enumerate(solution):
            key = (self.shifts[i], nurse_int)
            req_skill = self.req_skills[i]
            
            # penalidade de Skill
            skill = self.nurse_skills[key]
            if req_skill > skill:
                skill_deficit += (req_skill - skill)
                
            # acumulação de Workload
            nurse_workloads[key] = nurse_workloads.get(key, 0.0) + self.workloads[i]
            
        # penalidade de Excesso de Trabalho
        excess_workload = sum(
            (total_load - self.nurse_max_loads[key]) 
            for key, total_load in nurse_workloads.items() 
            if total_load > self.nurse_max_loads[key]
        )
                
        total_cost = (self.w_skill * skill_deficit) + (self.w_work * excess_workload)
        return -total_cost

    def optimize(self) -> pd.DataFrame:
        """Configura e executa a otimização evolutiva."""
        ga_instance = pygad.GA(
            num_generations=self.generations,
            num_parents_mating=int(self.pop_size * 0.2), # 20% da população
            fitness_func=self._fitness_function,
            sol_per_pop=self.pop_size,
            num_genes=self.num_tasks,
            gene_type=int,
            gene_space=self.gene_space,
            mutation_type="random",
            mutation_probability=self.mutation_rate,
            crossover_type="uniform",
            random_seed=self.seed,
            suppress_warnings=True,
            save_best_solutions=True
        )
        
        ga_instance.run()
        
        # salvar histórico de convergência
        history_dir = Path("results/convergence")
        history_dir.mkdir(parents=True, exist_ok=True)
        cost_history = [-val for val in ga_instance.best_solutions_fitness]
        
        file_path = history_dir / f"hist_{self.inst.instance_id}_seed{self.seed}_mut{self.mutation_rate}.json"
        with open(file_path, "w") as f:
            json.dump(cost_history, f)
        
        # reconstruir DataFrame de resposta
        solution, _, _ = ga_instance.best_solution()
        df_best = self.tasks_df[["global_shift", "room_id"]].copy()
        df_best["nurse_id"] = [self.nurse_int_to_id[int(n_int)] for n_int in solution]
        
        return df_best

# suíte de experimentos
def ga_solver(inst: InstanceData, seed: int, **kwargs) -> pd.DataFrame:
    optimizer = GeneticAlgorithmOptimizer(inst, seed, **kwargs)
    return optimizer.optimize()