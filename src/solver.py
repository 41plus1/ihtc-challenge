import pandas as pd
import random
import pulp

from src.data_loader import InstanceData
from src.experiments import evaluate_solution
from src.genetic_algorithm import ga_solver

class FixAndOptimizeNRA:
    """
    Abordagem Matheuristic: Fix & Optimize baseada em Programação Linear Inteira (PLI).
    """
    def __init__(self, instance_data: InstanceData, fitness_function, block_size: int = 5, max_iterations: int = 100):
        self.data = instance_data
        self.evaluate = fitness_function
        self.block_size = block_size
        self.max_iterations = max_iterations
        
    def optimize(self, initial_solution: pd.DataFrame) -> pd.DataFrame:
        current_solution = initial_solution.copy()
        current_cost = self.evaluate(current_solution)
        
        nurses_data = self.data.nurse_shifts['nurse_id'].unique().tolist()
        
        for i in range(self.max_iterations):
            len_bloc = min(self.block_size, len(nurses_data))
            nurses = random.sample(nurses_data, len_bloc)
            
            solution = self.exact_subproblem_solver(current_solution, nurses)
            cost = self.evaluate(solution)
            
            if cost < current_cost:
                current_solution = solution
                current_cost = cost
                
        return current_solution

    def exact_subproblem_solver(self, base_solution: pd.DataFrame, nurses: list) -> pd.DataFrame:
        occ = self.data.occupied_room_shifts
        nurse = self.data.nurse_shifts

        comb = pd.merge(
            occ[['global_shift', 'room_id', 'total_room_workload', 'max_skill_required']],
            nurse[['global_shift', 'nurse_id', 'skill_level', 'max_load']],
            on='global_shift'
        )

        comb['deficit'] = (comb['max_skill_required'] - comb['skill_level']).clip(lower=0)
        valid_tuples = list(zip(comb.nurse_id, comb.room_id, comb.global_shift))

        model = pulp.LpProblem("Subproblema_NRA", pulp.LpMinimize)
        x = pulp.LpVariable.dicts("x", valid_tuples, cat=pulp.LpBinary)
        
        keys = list(zip(nurse.nurse_id, nurse.global_shift))
        excess = pulp.LpVariable.dicts("excess", keys, lowBound=0, cat=pulp.LpInteger)

        w_skill = float(self.data.info.get("weights", {}).get("S2_room_nurse_skill", 1.0))
        w_work = float(self.data.info.get("weights", {}).get("S4_nurse_excessive_workload", 1.0))

        deficit_map = {(r.nurse_id, r.room_id, r.global_shift): r.deficit for r in comb.itertuples()}
        skill_cost = pulp.lpSum(x[key] * deficit_map[key] for key in valid_tuples if key in x)
        excess_cost = pulp.lpSum(excess.values())

        model += (w_work * excess_cost) + (w_skill * skill_cost), "Minimizar_Custo_Total"

        for (r, t), group in comb.groupby(['room_id', 'global_shift']):
            avaiable_nurses = group['nurse_id'].tolist()
            var = [x[(n, r, t)] for n in avaiable_nurses if (n, r, t) in x]
            if var:
                model += pulp.lpSum(var) == 1, f"Cobrir_Quarto_{r}_Turno_{t}"

        for (n, t), group in comb.groupby(['nurse_id', 'global_shift']):
            workload = pulp.lpSum(
                x[(n, row['room_id'], t)] * row['total_room_workload'] 
                for _, row in group.iterrows() if (n, row['room_id'], t) in x
            )
            max_load_n = group['max_load'].iloc[0]
            model += workload - max_load_n <= excess[(n, t)], f"Calc_Excesso_{n}_{t}"

        allocations = set(zip(base_solution.nurse_id, base_solution.room_id, base_solution.global_shift))

        for (n, r, t) in x.keys():
            if n not in nurses:
                if (n, r, t) in allocations:
                    model += x[(n, r, t)] == 1, f"Fix_On_{n}_{r}_{t}"
                else:
                    model += x[(n, r, t)] == 0, f"Fix_Off_{n}_{r}_{t}"

        model.solve(pulp.PULP_CBC_CMD(msg=False))

        new_allocations = []
        for (n, r, t), var in x.items():
            if var.varValue is not None and var.varValue > 0.5:
                new_allocations.append({'global_shift': t, 'room_id': r, 'nurse_id': n})

        return pd.DataFrame(new_allocations)

def pli_solver(inst: InstanceData, seed: int) -> pd.DataFrame:
    random.seed(seed)
    initial_solution = ga_solver(inst, seed)
    
    def fitness_function_adapted(solucao: pd.DataFrame) -> float:
        breakdown = evaluate_solution(inst, solucao)
        return breakdown.total
        
    solver = FixAndOptimizeNRA(instance_data=inst, fitness_function=fitness_function_adapted)
    return solver.optimize(initial_solution)