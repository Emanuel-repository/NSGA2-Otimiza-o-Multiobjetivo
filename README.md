# Otimização Logística: Seleção de Locais com NSGA-II 

Este projeto utiliza **Algoritmos Genéticos Multiobjetivo** para resolver o problema estratégico de seleção de locais para a instalação de filiais de distribuição em um cenário urbano.

##  Sobre o Projeto
A escolha de onde abrir uma nova unidade não é apenas uma decisão de custo, mas um equilíbrio entre eficiência operacional e nível de serviço. Este script em Python modela esse desafio como um problema combinatório complexo, otimizando três métricas simultaneamente:

1.  **Minimizar Custo:** Redução de gastos com aluguel/compra e operação.
2.  **Minimizar Tempo:** Redução da distância média de entrega para os clientes.
3.  **Maximizar Cobertura:** Atendimento do maior número de clientes dentro de um raio ideal.

## O Algoritmo: NSGA-II
Foi utilizado o **Non-dominated Sorting Genetic Algorithm II (NSGA-II)**, o padrão ouro para otimização multiobjetivo. O algoritmo se destaca por:
* **Dominância de Pareto:** Classifica as soluções em camadas de eficiência.
* **Crowding Distance:** Garante a diversidade das soluções encontradas.
* **Elitismo:** Preserva as melhores soluções ao longo das gerações.

## Tecnologias Utilizadas
* **Python 3**
* **Pymoo:** Framework especializado em otimização multiobjetivo.
* **Numpy:** Processamento vetorial de alta performance.
* **Folium:** Geração de mapas interativos para visualização espacial.
* **Matplotlib:** Visualização da Fronteira de Pareto.

## Resultados e Visualização
O projeto gera duas saídas principais:
1.  **Gráfico da Fronteira de Pareto:** Exibe os *trade-offs* entre custo e tempo, permitindo que o gestor visualize que o menor tempo de entrega exige maior investimento.
2.  **Mapa Interativo (HTML):** Uma visualização geográfica real com marcadores vermelhos para as filiais escolhidas e pontos azuis para os clientes atendidos.

## Como Usar
1. Instale as dependências:
   ```bash
   pip install -r requirements.txt

2.  Execute o script principal:
```bash
    python main.py