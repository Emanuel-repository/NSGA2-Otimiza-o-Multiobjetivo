import numpy as np
import matplotlib.pyplot as plt
import folium

# --- Imports do Pymoo (Core da Otimização) ---
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation # Corrigido: 'f' minúsculo
from pymoo.termination import get_termination

# =============================================================================
# 1. CLASSE DE DADOS (SIMULAÇÃO DO CENÁRIO REAL)
# =============================================================================
class CityData:
    """
    Gera dados sintéticos representando uma cidade:
    - Locais candidatos para filiais (Lat/Long)
    - Clientes distribuídos na cidade (Lat/Long)
    - Custos de instalação de cada local
    """
    def __init__(self, n_candidates=30, n_clients=200):
        np.random.seed(42) # Semente para resultados reprodutíveis
        
        # Simula coordenadas em uma região (ex: São Paulo)
        # Candidatos a Filial (Locais disponíveis para aluguel)
        self.locations = np.random.rand(n_candidates, 2) * 0.1 + [-16.73, -43.86]
        
        # Clientes espalhados pela cidade
        self.clients = np.random.rand(n_clients, 2) * 0.1 + [-16.73, -43.86]
        
        # Custo de Instalação (R$) para cada local candidato (entre 50k e 500k)
        self.setup_costs = np.random.randint(50, 500, size=n_candidates) * 1000

    def get_distance_matrix(self):
        """
        Calcula a matriz de distâncias entre TODOS os clientes e TODOS os locais candidatos.
        Retorna: Matriz (n_clientes x n_candidatos)
        """
        # Usando distância Euclidiana simples para o exemplo.
        # Em produção, usaria Haversine ou API de rotas.
        dists = np.zeros((len(self.clients), len(self.locations)))
        for i, client in enumerate(self.clients):
            for j, loc in enumerate(self.locations):
                dists[i, j] = np.linalg.norm(client - loc)
        return dists

# Inicializa os dados globais
DATA = CityData()
DIST_MATRIX = DATA.get_distance_matrix()


# =============================================================================
# 2. DEFINIÇÃO DO PROBLEMA DE OTIMIZAÇÃO (MODELAGEM)
# =============================================================================
class DistributionProblem(ElementwiseProblem):
    """
    Define a função de avaliação (Fitness Function).
    O algoritmo NSGA-II chamará esta classe milhares de vezes para testar soluções.
    """
    def __init__(self, max_branches=5):
        super().__init__(
            n_var=len(DATA.locations), # Número de variáveis = nº de locais candidatos
            n_obj=3,                   # 3 Objetivos: Custo, Tempo, Cobertura
            n_ieq_constr=1,            # 1 Restrição de desigualdade (Max Filiais)
            xl=0, xu=1,                # Limites das variáveis (0 ou 1)
            vtype=bool                 # Tipo da variável: Booleana (Binária)
        )
        self.max_branches = max_branches

    def _evaluate(self, x, out, *args, **kwargs):
        """
        x: Vetor booleano [True, False, True...] indicando quais locais foram escolhidos.
        """
        # Pega os índices onde x é True (locais onde abriremos filial)
        selected_indices = np.where(x)[0]
        
        # --- Tratamento de Caso Inválido (Nenhuma filial selecionada) ---
        if len(selected_indices) == 0:
            # Penaliza absurdamente para o algoritmo descartar essa solução
            out["F"] = [1e9, 1e9, 1e9]
            out["G"] = [1e9]
            return

        # === OBJETIVO 1: MINIMIZAR CUSTO TOTAL ===
        # Soma dos custos dos locais selecionados
        f1 = np.sum(DATA.setup_costs[selected_indices])

        # Cálculos auxiliares para Tempo e Cobertura
        # Pega apenas as distâncias dos locais selecionados
        dists_to_selected = DIST_MATRIX[:, selected_indices]
        # Para cada cliente, qual a distância da filial MAIS PRÓXIMA?
        min_dists = np.min(dists_to_selected, axis=1)

        # === OBJETIVO 2: MINIMIZAR TEMPO MÉDIO DE ENTREGA ===
        # Média das distâncias mínimas
        f2 = np.mean(min_dists)

        # === OBJETIVO 3: MAXIMIZAR COBERTURA (Minimizar não cobertos) ===
        # Consideramos coberto se distância < raio (ex: 0.02 graus ~ 2km)
        radius = 0.02
        # Quantos clientes estão fora do raio? (Queremos que isso seja 0)
        uncovered_count = np.sum(min_dists > radius)
        f3 = uncovered_count 

        # === RESTRIÇÃO ===
        # Número de filiais selecionadas - Máximo permitido <= 0
        # Ex: Se escolheu 7 e max é 5 -> 7 - 5 = 2 (Violação positiva)
        g1 = len(selected_indices) - self.max_branches

        # Salva resultados
        out["F"] = [f1, f2, f3] # Vetor de Objetivos
        out["G"] = [g1]         # Vetor de Restrições


# =============================================================================
# 3. CONFIGURAÇÃO E EXECUÇÃO DO ALGORITMO (NSGA-II)
# =============================================================================

# Configuração do Algoritmo Genético
algorithm = NSGA2(
    pop_size=100,  # Tamanho da população (número de soluções evoluindo juntas)
    sampling=BinaryRandomSampling(), # Gera população inicial aleatória binária
    crossover=TwoPointCrossover(prob=0.9), # Cruza genes de duas soluções (Pais)
    mutation=BitflipMutation(prob=0.05),   # Muta bits aleatórios (0 vira 1, 1 vira 0)
    eliminate_duplicates=True # Remove gêmeos para manter diversidade genética
)

# Critério de Parada
termination = get_termination("n_gen", 100) # Rodar por 100 gerações

print("Iniciando otimização com NSGA-II...")
problem = DistributionProblem(max_branches=5)

# Executa a otimização
res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               save_history=False,
               verbose=True) # Verbose=True mostra o progresso no terminal

print(f"Otimização concluída em {res.exec_time:.2f} segundos.")
print(f"Soluções ótimas encontradas (Fronteira de Pareto): {len(res.F)}")


# =============================================================================
# 4. EXPORTAÇÃO E VISUALIZAÇÃO
# =============================================================================

# --- A. Salvar Gráfico da Fronteira de Pareto ---
# Eixo X: Custo, Eixo Y: Tempo, Cor: Clientes Não Cobertos
plt.figure(figsize=(10, 6))
sc = plt.scatter(res.F[:, 0], res.F[:, 1], c=res.F[:, 2], cmap='viridis', s=60, edgecolors='k')
plt.colorbar(sc, label='Número de Clientes NÃO Cobertos')
plt.xlabel('Custo Total de Instalação (R$)')
plt.ylabel('Distância Média (Proxy de Tempo)')
plt.title('Fronteira de Pareto: Trade-offs da Logística')
plt.grid(True, alpha=0.3)
plt.savefig("meu_grafico_pareto.png") # Salva imagem para o slide
print("Gráfico salvo como 'meu_grafico_pareto.png'")


# --- B. Salvar Mapa Interativo (Escolhendo uma solução mediana) ---
# Ordena soluções pelo custo e pega a do meio (trade-off balanceado)
sorted_indices = np.argsort(res.F[:, 0])
median_idx = sorted_indices[len(sorted_indices)//2]

best_solution_vector = res.X[median_idx] # Vetor binário da solução escolhida
selected_locs = DATA.locations[best_solution_vector] # Coordenadas reais

# Cria mapa centralizado na média dos pontos
center_lat = np.mean(DATA.locations[:, 0])
center_long = np.mean(DATA.locations[:, 1])
m = folium.Map(location=[center_lat, center_long], zoom_start=13)

# 1. Plota TODOS os Clientes (Pequenos pontos azuis)
for client in DATA.clients:
    folium.CircleMarker(
        location=client, radius=2, color='blue', fill=True, fill_opacity=0.6
    ).add_to(m)

# 2. Plota Locais SELECIONADOS pelo Algoritmo (Marcadores Vermelhos)
for i, loc in enumerate(selected_locs):
    folium.Marker(
        location=loc,
        popup=f"<b>Filial Nova {i+1}</b>",
        icon=folium.Icon(color='red', icon='warehouse', prefix='fa')
    ).add_to(m)

m.save("meu_mapa_otimizado.html") # Salva arquivo HTML
print("Mapa salvo como 'meu_mapa_otimizado.html'")