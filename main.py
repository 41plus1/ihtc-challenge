from pathlib import Path
from src.experiments import run_suite, greedy_baseline
from src.metaheuristic import ga_solver

def main():
    print("Iniciando bateria de testes IHTC-2024 (NRA)...")

    # caminhos
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"
    out_dir = base_dir / "results"

    # dicionário com os solvers que serão avaliados
    methods_to_test = {
        "Greedy baseline": greedy_baseline,
        "Genetic Algorithm": ga_solver
    }

    # executando a suite
    df_results = run_suite(
        data_dir = data_dir,
        instance_ids = ["i04", "i06"],
        methods = methods_to_test,
        repeats = 6,
        base_seed = 42,
        out_dir = out_dir
    )

    print("\nResumo dos resultados:")
    resumo = df_results.groupby(["instance_id", "method"])["objective"].agg(["min", "mean", "std"])
    print(resumo)

if __name__ == "__main__":
    main()
