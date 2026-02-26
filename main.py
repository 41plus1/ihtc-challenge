import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.experiments import run_suite, greedy_baseline
from src.solver import pli_solver
from src.genetic_algorithm import ga_solver

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

def plot_boxplots(df):
    """Gera um boxplot comparando a distribuição de custos por método."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="instance_id", y="objective", hue="method", palette="Set2")
    plt.title("Distribuição do Custo (Objetivo) por Instância e Método")
    plt.ylabel("Custo (Penalidades)")
    plt.xlabel("Instância")
    plt.tight_layout()
    plt.savefig("results/plot_boxplot_custos.png", dpi=300)
    plt.close()

def plot_runtime(df):
    """Gera um gráfico comparando os tempos médios de execução."""
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="instance_id", y="runtime_s", hue="method", palette="pastel", errorbar=None)
    plt.title("Tempo Médio de Execução por Método")
    plt.ylabel("Tempo (Segundos)")
    plt.xlabel("Instância")
    plt.tight_layout()
    plt.savefig("results/plot_tempo_execucao.png", dpi=300)
    plt.close()

def plot_convergence():
    """Lê os JSONs da Meta-heurística e gera as curvas de convergência."""
    history_dir = Path("results/convergence")
    files = list(history_dir.glob("hist_i06_*_mut0.15.json")) 
    
    if not files: return
        
    plt.figure(figsize=(10, 6))
    for file in files:
        with open(file, 'r') as f:
            history = json.load(f)
            plt.plot(history, alpha=0.6, linewidth=2)

    plt.title("Convergência do Algoritmo Genético (Múltiplas Seeds) - Instância i06")
    plt.xlabel("Gerações")
    plt.ylabel("Custo")
    plt.tight_layout()
    plt.savefig("results/plot_convergencia_i06.png", dpi=300)
    plt.close()

def run_sensitivity_analysis(data_dir, out_dir):
    """Executa a análise de sensibilidade variando a taxa de mutação."""
    print("\n--- Iniciando Análise de Sensibilidade (Taxa de Mutação) ---")
    mutations = [0.01, 0.15, 0.50]
    plt.figure(figsize=(10, 6))
    
    from src.data_loader import load_instance
    inst = load_instance(data_dir=data_dir, instance_id="i04")
    
    for mut in mutations:
        ga_solver(inst, seed=42, mutation_rate=mut)
        
        # carregar o ficheiro gerado
        file = Path(f"results/convergence/hist_i04_seed42_mut{mut}.json")
        if file.exists():
            with open(file, 'r') as f:
                history = json.load(f)
                plt.plot(history, label=f"Mutação = {mut*100}%", linewidth=2.5)
                
    plt.title("Análise de Sensibilidade: Impacto da Taxa de Mutação (Instância i04)")
    plt.xlabel("Gerações")
    plt.ylabel("Custo")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/plot_sensibilidade_mutacao.png", dpi=300)
    plt.close()
    print("Gráfico de sensibilidade salvo em 'results/plot_sensibilidade_mutacao.png'.")

def main():
    print("Iniciando bateria de testes IHTC-2024...")

    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"
    out_dir = base_dir / "results"
    out_dir.mkdir(exist_ok=True)

    methods_to_test = {
        "Greedy Baseline": greedy_baseline,
        "Genetic Algorithm": ga_solver,
        "Solver PLI": pli_solver
    }

    # run principal
    df_results = run_suite(
        data_dir = data_dir,
        instance_ids = ["i04", "i06"],
        methods = methods_to_test,
        repeats = 5, 
        base_seed = 42,
        out_dir = out_dir
    )

    print("\nResumo Final:")
    print(df_results.groupby(["instance_id", "method"])["objective"].agg(["min", "mean", "std"]))

    # gerar os gráficos 
    plot_boxplots(df_results)
    plot_runtime(df_results)
    plot_convergence()
    
    # rodar e gerar gráfico de sensibilidade
    run_sensitivity_analysis(data_dir, out_dir)
    print("Testes e geração de gráficos concluídos com sucesso!")

if __name__ == "__main__":
    main()