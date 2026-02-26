# IHTC-2024: Nurse-to-Room Assignment (NRA) Optimization

Repositório oficial do Projeto Final da disciplina de Matemática Computacional. 
O objetivo deste projeto é otimizar a alocação de enfermeiros a quartos de hospital por turno, minimizando o défice de competências (*skill deficit*) e o excesso de carga de trabalho (*excessive workload*), com base no subproblema NRA da competição *Integrated Healthcare Timetabling Competition 2024* (IHTC-2024).

## Abordagens Implementadas

Para resolver este problema de otimização combinatória (NP-Difícil) e lidar com as rigorosas restrições de cobertura e disponibilidade, implementamos três abordagens:

1. **Heurística Gulosa (Greedy Baseline):** Atua como ponto de referência, alocando os quartos ao enfermeiro com maior nível de competência disponível no turno. Ignora o planeamento a longo prazo, servindo para demonstrar o impacto negativo da sobrecarga de trabalho.
2. **Algoritmo Genético (PyGAD):** Uma meta-heurística focada na exploração global do espaço de busca. Utiliza a funcionalidade de `gene_space` para garantir a geração de cromossomas 100% factíveis, eliminando a necessidade de funções pesadas de reparação.
3. **Matheurística - Fix & Optimize (Solver PLI):** Utiliza o Algoritmo Genético para obter uma solução global viável inicial e aplica Programação Linear Inteira (PLI) via `PuLP/CBC` para otimizar exata e iterativamente pequenos blocos (vizinhanças) de 5 enfermeiros. O modelo matemático utiliza estritamente variáveis binárias e inteiras.

## Bibliotecas Utilizadas

O projeto foi desenvolvido em Python e apoia-se num conjunto robusto de bibliotecas científicas e de otimização:

* **`pandas`** e **`numpy`**: Utilizados para a manipulação rápida de dados, pré-processamento das instâncias (ficheiros CSV/JSON) e cálculos vetorizados de matrizes de custo.
* **`pygad`**: Biblioteca para a construção do Algoritmo Genético, responsável por gerir a população, cruzamentos, mutações e o espaço de genes limitante (*gene space*).
* **`pulp`**: Interface de modelagem para Programação Linear Inteira (PLI). Utiliza o solver *CBC* (integrado) para encontrar soluções exatas para os subproblemas matemáticos da matheurística.
* **`matplotlib`** e **`seaborn`**: Utilizados em conjunto para a renderização de gráficos, essenciais para a análise estatística (boxplots, curvas de convergência e tempos de execução).

## Estrutura do Projeto

```text
ihtc-challenge/
├── data/                       # Instâncias do IHTC-2024 (ex: i04, i06)
├── results/                    # Resultados das execuções (CSVs, JSONs e Gráficos de Análise)
├── src/
│   ├── data_loader.py          # Parser e carregamento das instâncias (InstanceData)
│   ├── experiments.py          # Motor de avaliação (função objetivo) e suíte de repetições
│   ├── genetic_algorithm.py    # Implementação do Algoritmo Genético (PyGAD)
│   └── solver.py               # Modelagem Matemática PLI (Fix & Optimize com PuLP)
├── main.py                     # Orquestra os testes oficiais e gera os gráficos
├── main.ipynb                  # Notebook para apresentação visual dos resultados
├── pyproject.toml / uv.lock    # Ficheiros de configuração do ambiente e dependências
└── README.md                   # Documentação do projeto
