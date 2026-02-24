import pandas as pd
import random
import pulp

from src.data_loader import InstanceData
from src.experiments import evaluate_solution, greedy_baseline

class FixAndOptimizeNRA:
    def __init__(self, instance_data: InstanceData, fitness_function):
        self.data = instance_data
        self.evaluate = fitness_function
        
    def optimize(self, initial_solution: pd.DataFrame, max_it: int = 20) -> pd.DataFrame:
        current_solution = initial_solution.copy()
        current_cost = self.evaluate(current_solution)
        
        print(f"Custo Inicial: {current_cost}")
        
        # pega a lista de todos os enfermeiros disponíveis na instância
        nurses_data = self.data.nurse_shifts['nurse_id'].unique().tolist()
        
        for i in range(max_it):
            # define a vizinhança
            len_bloc = min(3, len(nurses_data))
            nurses = random.sample(nurses_data, len_bloc)
            
            # otimizar subproblema
            solution = self.exact_subproblem_solver(current_solution, nurses)
            
            # criterio de avaliacao
            cost = self.evaluate(solution)
            
            if cost < current_cost:
                print(f"Iteração {i} | Custo melhorou: {current_cost} -> {cost} | Enfermeiros alterados: {nurses}")
                current_solution = solution
                current_cost = cost
                
        print(f"Otimização finalizada. Custo Final: {current_cost}")
        return current_solution

    def exact_subproblem_solver(self, base_solution: pd.DataFrame, nurses: list) -> pd.DataFrame:
        occ = self.data.occupied_room_shifts
        nurse = self.data.nurse_shifts

        comb = pd.merge(
            occ[['global_shift', 'room_id', 'total_room_workload', 'max_skill_required']],
            nurse[['global_shift', 'nurse_id', 'skill_level', 'max_load']],
            on='global_shift'
        )

        # calcula o déficit de skill
        comb['deficit'] = (comb['max_skill_required'] - comb['skill_level']).clip(lower=0)

        valid_tuples = list(zip(comb.nurse_id, comb.room_id, comb.global_shift))

        # modelo
        model = pulp.LpProblem("Subproblema_NRA", pulp.LpMinimize)

        # variaveis de decisao
        x = pulp.LpVariable.dicts("x", valid_tuples, cat=pulp.LpBinary)
        
        keys = list(zip(nurse.nurse_id, nurse.global_shift))
        excess = pulp.LpVariable.dicts("excess", keys, lowBound=0, cat=pulp.LpContinuous)

        # funcao objetivo
        w_skill = float(self.data.info.get("weights", {}).get("S2_room_nurse_skill", 1.0))
        w_work = float(self.data.info.get("weights", {}).get("S4_nurse_excessive_workload", 1.0))

        deficit_map = {(r.nurse_id, r.room_id, r.global_shift): r.deficit for r in comb.itertuples()}

        skill_cost = pulp.lpSum(x[key] * deficit_map[key] for key in valid_tuples if key in x)
        excess_cost = pulp.lpSum(excess.values())

        model += (w_work * excess_cost) + (w_skill * skill_cost), "Minimizar_Custo_Total"

        # restricoes
        # 1 enfermeiro por quarto/turno
        for (r, t), group in comb.groupby(['room_id', 'global_shift']):
            avaiable_nurses = group['nurse_id'].tolist()
            var = [x[(n, r, t)] for n in avaiable_nurses if (n, r, t) in x]
            if var:
                model += pulp.lpSum(var) == 1, f"Cobrir_Quarto_{r}_Turno_{t}"

        # workload
        for (n, t), group in comb.groupby(['nurse_id', 'global_shift']):
            workload = pulp.lpSum(
                x[(n, row['room_id'], t)] * row['total_room_workload'] 
                for _, row in group.iterrows() if (n, row['room_id'], t) in x
            )
            max_load_n = group['max_load'].iloc[0]
            model += workload - max_load_n <= excess[(n, t)], f"Calc_Excesso_{n}_{t}"

        # congelamento
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
                new_allocations.append({
                    'global_shift': t, 
                    'room_id': r, 
                    'nurse_id': n
                })

        return pd.DataFrame(new_allocations)


def fix_and_optimize_solver(inst: InstanceData, seed: int) -> pd.DataFrame:
    random.seed(seed)
    
    initial_solution = greedy_baseline(inst, seed)
    
    def fitness_function_adapted(solucao: pd.DataFrame) -> float:
        breakdown = evaluate_solution(inst, solucao)
        return breakdown.total
        
    solver = FixAndOptimizeNRA(
        instance_data=inst, 
        fitness_function_colega=fitness_function_adapted
    )
    
    final_solution = solver.optimize(initial_solution, max_it=20)
    
    return final_solution