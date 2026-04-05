# Relatório de Análise Exploratória: Lesões no Futebol Europeu (2020–2025)

## Conhecendo os dados

Nesta etapa, realizamos uma imersão nos dados para compreender sua estrutura e qualidade. [cite_start]O conjunto de dados original contém registros de lesões e afastamentos de jogadores das cinco principais ligas europeias (Bundesliga, Premier League, La Liga, Ligue 1 e Serie A)[cite: 1976].

### Estrutura da Base
A base de dados é composta por **15.603 registros** e **11 atributos**. Para facilitar a análise, realizamos a tradução das colunas para o português:

* [cite_start]`temporada`: Ciclo anual da ocorrência (ex: 20/21)[cite: 1976].
* [cite_start]`tipo_lesao`: Descrição da condição médica ou motivo do afastamento[cite: 1976].
* `duracao_dias`: Tempo total de afastamento (convertido para numérico para análise).
* [cite_start]`jogos_perdidos`: Quantidade de partidas oficiais que o atleta não pôde atuar[cite: 1976].
* [cite_start]`data_inicio_lesao` / `data_fim_lesao`: Período cronológico do evento[cite: 1976].
* [cite_start]`jogador` / `idade` / `posicao`: Dados demográficos e táticos do atleta[cite: 1976].
* [cite_start]`clube` / `liga`: Contexto esportivo[cite: 1976].

### Qualidade dos Dados
A análise inicial de integridade revelou dados extremamente consistentes:
* **Dados Ausentes:** 0% de valores faltantes em todas as colunas.
* **Duplicatas:** Nenhum registro duplicado identificado.
* **Consistência Temporal:** O período coberto vai de fevereiro de 2020 a janeiro de 2026, com lógica de datas validada (data de fim sempre posterior à de início).

### Estatísticas Descritivas e Visualizações
Utilizamos Python para calcular as métricas de centralidade e dispersão das variáveis quantitativas:

```python
# Seleção das variáveis númericas para análise de centralidade e variabilidade
cols_num = ['idade', 'duracao_dias', 'jogos_perdidos']

# Cálculo de estatísticas detalhadas
estatistica = df_injuries[cols_num].describe().T
estatistica['mediana'] = df_injuries[cols_num].median()
estatistica['moda'] = df_injuries[cols_num].mode().iloc[0]
estatistica['variancia'] = df_injuries[cols_num].var()
estatistica['amplitude'] = df_injuries[cols_num].max() - df_injuries[cols_num].min()

# Renomeando para visualização clara
estatistica = estatistica[['count', 'mean', 'mediana', 'moda', 'std', 'variancia', 'min', 'max', 'amplitude']]

display(estatistica)
```
A tabela abaixo apresenta as medidas de tendência central e dispersão para as três variáveis numéricas do dataset.

| Variável | Count | Média | Mediana | Moda | Desvio Padrão | Variância | Mín | Máx | Amplitude |
|---|---|---|---|---|---|---|---|---|---|
| `idade` | 15.603 | 26,55 | 26,0 | 25 | 4,40 | 19,34 | 16 | 43 | 27 |
| `duracao_dias` | 15.603 | 36,10 | 18,0 | 8 | 54,42 | 2.961,52 | 1 | 1.013 | 1.012 |
| `jogos_perdidos` | 15.603 | 5,51 | 3,0 | 1 | 7,64 | 58,30 | 1 | 145 | 144 |

### Análise

Os dados revelam padrões importantes sobre o perfil das lesões no futebol europeu:

- **Idade:** Os atletas lesionados têm em média **26,5 anos**, com mediana de 26 e moda de 25, indicando uma distribuição relativamente simétrica e concentrada na faixa de atletas jovens-adultos em plena carreira. A amplitude de 27 anos (de 16 a 43) mostra que lesões ocorrem em todas as fases da carreira profissional.

- **Duração das Lesões (`duracao_dias`):** Esta é a variável com maior assimetria. A **média de 36 dias é o dobro da mediana de 18 dias**, e a moda é de apenas 8 dias. Isso confirma uma distribuição com forte cauda longa à direita: a grande maioria das lesões é de curta duração, mas poucos eventos graves (chegando a 1.013 dias) puxam a média para cima. A variância elevada (2.961,52) reforça a alta heterogeneidade no tempo de recuperação.

- **Jogos Perdidos:** Segue o mesmo padrão assimétrico de `duracao_dias`. A **média de 5,5 jogos é quase o dobro da mediana de 3**, e a moda é 1, indicando que a maioria dos atletas perde apenas 1 jogo por evento de lesão. Casos extremos (até 145 jogos) representam lesões gravíssimas que distorcem a média.

> **Conclusão:** Para análises comparativas e modelos preditivos, recomenda-se priorizar a **mediana** como medida de centralidade para `duracao_dias` e `jogos_perdidos`, dado o forte efeito dos outliers sobre a média. Técnicas como winsorização ou transformação logarítmica devem ser consideradas nas etapas seguintes.

*Visualizações inseridas:*
* **Box Plots:** Foram gerados para identificar a severidade das lesões por posição do jogador e idade, permitindo a visualização clara de *outliers* (lesões de longa duração, como rupturas de ligamento cruzado que chegam a +200 dias).

<img width="1005" height="942" alt="output" src="https://github.com/user-attachments/assets/16b0c42e-a20f-4207-8496-d9f2266c59c9" />

<img width="1005" height="864" alt="output2" src="https://github.com/user-attachments/assets/eb0c85c2-7f29-4fa3-99a3-8fa032dad881" />


* **Histogramas:** [Espaço para descrever a distribuição da frequência das lesões].

---

Para complementar a análise estatística descritiva, foram utilizados **histogramas** e **boxplots**, com o objetivo de representar visualmente a distribuição da variável `duracao_dias` e comparar seu comportamento entre as diferentes ligas europeias.

O **histograma da duração das lesões** evidencia uma distribuição fortemente assimétrica à direita. Observa-se uma grande concentração de casos em afastamentos curtos, especialmente nas primeiras faixas de dias, enquanto a frequência diminui rapidamente à medida que a duração aumenta. Esse comportamento confirma o que já havia sido identificado nas medidas de tendência central: a maior parte das lesões possui curta duração, mas há um número reduzido de casos extremamente longos que estendem a cauda da distribuição. Assim, trata-se de uma variável com forte presença de outliers e alta dispersão.

Já o **boxplot da duração das lesões por liga** permite comparar a distribuição dos afastamentos entre Bundesliga, Premier League, La Liga, Ligue 1 e Serie A. De modo geral, as medianas e os intervalos interquartis das ligas são relativamente semelhantes, sugerindo um comportamento central próximo entre os campeonatos. No entanto, todas as ligas apresentam quantidade expressiva de valores extremos, indicando que lesões graves ou afastamentos muito longos são um fenômeno recorrente em todo o futebol europeu, e não restrito a uma competição específica.

Além disso, nota-se que algumas ligas apresentam outliers mais elevados do que outras, como a Premier League e a La Liga, que exibem casos de afastamentos excepcionalmente longos. Isso sugere que, embora o padrão central seja semelhante, a severidade máxima das lesões pode variar entre as ligas.

Em conjunto, essas visualizações reforçam três pontos principais:
- a distribuição de `duracao_dias` não é normal, sendo marcada por forte assimetria positiva;
- a maioria das lesões é de curta duração;
- os casos extremos devem ser mantidos na base, pois representam eventos reais e relevantes para a compreensão da severidade das lesões.

Dessa forma, os gráficos confirmam que a variável `duracao_dias` exige atenção especial nas análises posteriores, sendo recomendável o uso de medidas robustas, como mediana e percentis, além de possíveis transformações ou segmentações em etapas preditivas futuras.

<img width="1093" height="365" alt="Captura de tela de 2026-04-04 18-11-56" src="https://github.com/user-attachments/assets/603e695a-52af-44f7-974a-974a6d76d4fd" />

---

* **Correlação e Dispersão:** Para investigar as relações existentes entre as variáveis numéricas do dataset, foram utilizados um **mapa de calor de correlação** e um **gráfico de dispersão**, técnicas complementares que permitem identificar tanto a intensidade quanto a forma das associações entre as variáveis.

  O **mapa de calor** apresenta os coeficientes de correlação de Pearson entre `idade`, `duracao_dias` e `jogos_perdidos`. Os resultados revelam dois padrões distintos:

- A correlação entre `duracao_dias` e `jogos_perdidos` é de **0,93**, indicando uma associação positiva muito forte. Isso significa que, quanto maior o tempo de afastamento de um atleta, maior tende a ser o número de partidas que ele perde, o que é esperado e confirma a coerência interna dos dados.
- Já a correlação entre `idade` e as demais variáveis é de apenas **-0,06** em ambos os casos, valor próximo de zero que indica ausência de relação linear relevante. Isso sugere que a idade do atleta, isoladamente, não é um fator determinante para a gravidade ou duração de uma lesão neste dataset.

O **gráfico de dispersão** entre `duracao_dias` e `jogos_perdidos` confirma visualmente a forte correlação identificada. A nuvem de pontos apresenta uma tendência linear clara e crescente: atletas com afastamentos mais longos tendem a perder mais jogos de forma proporcional. Observa-se também que a dispersão aumenta conforme a duração cresce, o que indica maior variabilidade nos casos mais graves, possivelmente influenciada pelo calendário de cada liga ou pela fase da temporada em que a lesão ocorreu.

Alguns pontos extremos são visíveis no canto superior direito do gráfico, representando os casos mais severos do dataset, com afastamentos superiores a 600 dias e mais de 100 jogos perdidos. Esses casos, embora raros, reforçam a importância de considerar a severidade das lesões como uma dimensão crítica em análises preditivas futuras.

Em síntese, os resultados desta etapa indicam que:
- O **tempo de afastamento é o principal preditor do impacto esportivo** (jogos perdidos);
- A **idade não apresenta correlação linear** com a gravidade das lesões neste conjunto de dados;
- A relação entre `duracao_dias` e `jogos_perdidos` é robusta e consistente, validando ambas as variáveis como representações complementares da severidade de uma lesão.

<img width="1097" height="334" alt="Captura de tela de 2026-04-01 20-54-01" src="https://github.com/user-attachments/assets/07bb3990-17c7-46f4-b4c7-5099273308df" />

---

* **Barras e Colunas:** O gráfico com o **Top 10 tipos de lesão mais frequentes** mostra que as ocorrências estão fortemente concentradas em problemas musculares. A **hamstring injury** aparece como a categoria mais frequente, seguida por registros como **corona virus**, **muscle injury** e **muscular problems**. Esse resultado sugere que lesões musculares e indisponibilidades físicas de curta ou média duração compõem grande parte dos eventos registrados no dataset. A presença de `corona virus` entre os tipos mais frequentes também indica que a base não se limita apenas a lesões traumáticas ou musculoesqueléticas, mas inclui outras condições de afastamento que impactaram o futebol europeu no período analisado. Já o gráfico de **média de dias de afastamento por posição** revela diferenças importantes entre funções táticas. Observa-se que posições como **goalkeeper**, **right midfield**, **centre-back** e **left-back** apresentam médias mais altas de dias afastados, enquanto posições como **midfielder** e **second striker** apresentam médias menores. Isso sugere que o impacto médio das lesões pode variar de acordo com as exigências físicas, o tipo de movimento mais frequente e o perfil biomecânico de cada posição.

<img width="1094" height="327" alt="Captura de tela de 2026-04-01 21-00-17" src="https://github.com/user-attachments/assets/cc1367dc-1b21-4241-934d-5d2a0083e3f0" />

---

## Descrição dos achados

A partir da EDA, os seguintes pontos foram identificados como cruciais para o futuro treinamento da IA:

1.  **Amplitude da Severidade:** As lesões variam de 1 dia a 1.013 dias de afastamento. A grande massa de dados concentra-se em lesões de curta duração, mas os *outliers* representam o maior impacto financeiro e técnico para os clubes.
2.  **Variabilidade por Posição:** [Espaço para preencher se atacantes se lesionam mais que zagueiros, conforme observado nos gráficos].
3.  **Correlações:**
    * **Forte:** Identificamos uma correlação positiva forte entre `duracao_dias` e `jogos_perdidos`.
    * **Moderada/Fraca:** [Espaço para descrever a relação entre idade e frequência de lesões].
4.  **Impacto Externo:** O dataset registra um volume significativo de afastamentos por "Corona virus" e "Quarantine" entre 2020 e 2022, o que deve ser tratado como um padrão atípico na engenharia de dados.


   ## 📊 Análise de Multicolinearidade (VIF)

Durante a preparação dos dados para a Análise Preditiva, foi realizado o cálculo do **VIF (Variance Inflation Factor)** para identificar possíveis problemas de multicolinearidade entre as variáveis numéricas do conjunto de dados.

### 🔍 Resultados Obtidos:
*   🚨 **`Games missed` (7.90) e `Days` (7.89):** Ambas apresentaram valores superiores a 5. Isso comprova matematicamente uma forte multicolinearidade, o que faz sentido no contexto do esporte: o tempo de afastamento em dias afeta diretamente e de forma proporcional a quantidade de partidas perdidas pelo jogador.
*   ✅ **`player_age` (1.00):** O VIF próximo de 1 indica ausência de colinearidade, mostrando que a idade do jogador é uma variável independente sólida em relação ao tempo de afastamento.

### 💡 Conclusão e Recomendação para a Modelagem:
> Ação: Devemos evitar o uso simultâneo de `Games missed` e `Days` como variáveis independentes no mesmo modelo de Machine Learning. Para não inflar a variância, causar viés e prejudicar as previsões, recomenda-se selecionar apenas uma dessas variáveis de impacto para treinar o algoritmo na próxima etapa.

## 🔍 Análise Exploratória de Dados (EDA) - Parte 2

Nesta etapa, focamos na preparação das variáveis categóricas e na visualização da distribuição de lesões para subsidiar o modelo preditivo.

### 1. Distribuição de Lesões por Posição e Liga
Foram gerados gráficos de barras para identificar padrões de incidência:
*   **Posição:** Analisamos quais setores do campo (Goleiros, Defensores, Meias ou Atacantes) sofrem mais afastamentos.
*   **Ligas:** Comparação do volume de dados entre as cinco grandes ligas europeias (Bundesliga, Premier League, La Liga, Ligue 1 e Serie A).

<img width="861" height="557" alt="download" src="https://github.com/user-attachments/assets/c8adbde7-ea22-46a4-89dd-78c9b90b849b" />

<img width="859" height="543" alt="download 2" src="https://github.com/user-attachments/assets/3551bf98-8eb3-460b-ad7e-3f814bf613e0" />

### 2. Codificação de Variáveis (Feature Engineering)
Como modelos de Machine Learning (como o Random Forest) operam apenas com dados numéricos, aplicamos a técnica de **One-Hot Encoding** utilizando a função `get_dummies`.

*   **O que foi feito:** As colunas de texto `player_position` e `league` foram transformadas em vetores binários (True/False).
*   **Resultado:** O dataset agora possui colunas específicas para cada categoria, permitindo que o algoritmo calcule pesos matemáticos para cada posição e liga sem criar uma hierarquia falsa entre elas.

### 3. Validação Final para Modelagem
Com as variáveis numéricas validadas pelo VIF e as categóricas devidamente codificadas, o dataset está pronto para a fase de **Treinamento e Teste**, garantindo que:
1. Não há redundância de dados (Multicolinearidade controlada).
2. Todas as informações táticas e de contexto estão em formato legível para a IA.
---

## Análises com Pairplot

### 1) Relações entre Variáveis Numéricas com Pairplot Balanceado (idade, duração das lesões e jogos perdidos por liga)

A proposta desta seção foi avaliar a relação entre as três variáris numéricas: idades dos jogadores, tempo de lesão (em dias) e número de jogos peridos. Além disso, buscamos comparar esses padrões entre as diferentes ligas europeias analisadas.

Para garantir uma comparação justa entre as ligas, foi selecionada uma amostragem que inclui 100 registros para cada liga. Também foi realizado um filtro para remover lesões que possuem duração supeerior a 200 dias, com o objetico de reduzir o impacto de valroes extremos.

<img width="905" height="770" alt="image" src="https://github.com/user-attachments/assets/d6f264cf-84db-49f2-8aa2-c4c9efb9a1c7" />

Conclusão:
O conjunto de gráficos evidencia que a duração da lesão é o principal fator associado ao número de jogos perdidos, apresentando uma forte relação positiva. Por outro lado, a idade não demonstra influência significativa sobre a gravidade ou o impacto das lesões. Além disso, observa-se que a maioria das lesões é de curta duração e baixo impacto, embora existam casos extremos. As ligas europeias apresentam comportamento semelhante, sem diferenças relevantes entre elas.

### 2) Análise por Posição dos Jogadores (Pairplot Balanceado)

Nesta etapa, foi realizada uma análise exploratória com o objetivo de investigar as relações entre as variáveis idade, duração das lesões (em dias) e número de jogos perdidos, considerando agora a posição dos jogadores em campo como fator de segmentação.

Para isso, foram selecionadas as cinco posições mais frequentes no conjunto de dados, garantindo maior representatividade na análise. Em seguida, foi aplicada uma amostragem balanceada, com a seleção de uma quantidade fixa de registros para cada posição, permitindo uma comparação mais justa entre os grupos.

Assim como na análise anterior, também foi aplicado um filtro para excluir lesões com duração superior a 200 dias, com o objetivo de reduzir a influência de valores extremos e facilitar a visualização dos padrões centrais.

Essa análise permite identificar possíveis diferenças no comportamento das lesões de acordo com a posição dos jogadores, contribuindo para uma compreensão mais aprofundada dos fatores associados à severidade e ao impacto das lesões no futebol.

<img width="1185" height="1208" alt="image" src="https://github.com/user-attachments/assets/98ecb614-f098-4fc9-9501-e13e2fb8aeae" />

Conclusão:
A análise do pairplot por posição indica que o padrão de lesões é bastante semelhante entre os diferentes papéis em campo. A duração da lesão mantém forte relação com o número de jogos perdidos em todas as posições, enquanto não há evidência de que alguma posição específica esteja associada a maior severidade das lesões. Além disso, observa-se que a maioria das lesões é de curta duração, com poucos casos extremos distribuídos entre todas as posições.

### 3) Análise da Severidade das Lesões e seu Impacto (Pairplot)

Nesta etapa, foi realizada uma preparação e visualização dos dados com o objetivo de analisar a severidade das lesões a partir de uma abordagem em categorias.

Inicialmente, foi criada uma nova variável denominada severidade, a partir da variável numérica duração das lesões (duracao_dias). Para isso, utilizamos a função pd.cut, que permite segmentar dados contínuos em intervalos. As lesões foram classificadas em três categorias: leve, para afastamentos de até 15 dias; moderada, para durações entre 16 e 60 dias; e grave, para períodos superiores a 60 dias. Essa transformação permitiu converter uma variável contínua em uma variável categórica, facilitando a análise comparativa entre níveis de gravidade.

Em seguida, foi realizada uma amostragem aleatória de 1000 registros do conjunto de dados, com o objetivo de reduzir o volume de informações e melhorar a visualização gráfica, mantendo a representatividade dos dados. O uso do parâmetro random_state=42 garantiu a repetibilidade da amostra, permitindo que os mesmos registros sejam selecionados em diferentes execuções.

<img width="868" height="770" alt="image" src="https://github.com/user-attachments/assets/15f611f5-3687-4dd7-b486-a5ab27d65115" />

Conclusão:
Os resultados demonstram que a severidade da lesão é um dos principais fatores para explicar o impacto esportivo, apresentando forte associação com a duração do afastamento e o número de jogos perdidos. A classificação adotada mostrou-se eficaz para segmentar os dados e evidenciar padrões relevantes. Por outro lado, a idade não apresenta influência significativa sobre a gravidade ou o impacto das lesões. Dessa forma, a variável severidade se destaca como uma importante dimensão para análises futuras.

## Ferramentas utilizadas

O projeto foi desenvolvido utilizando o ecossistema de Ciência de Dados da linguagem **Python**, com as seguintes bibliotecas:

* **Pandas:** Manipulação de dados, tratamento de tipos e tradução de colunas.
* **Matplotlib:** Geração e customização de gráficos, incluindo títulos, ajustes visuais e exibição das figuras.
* **Matplotlib:** Geração e customização de gráficos, incluindo títulos, ajustes visuais e exibição das figuras.
* **NumPy:** Suporte para cálculos matemáticos e operações em arrays.
* **Matplotlib / Seaborn:** Geração de gráficos estatísticos, box plots e histogramas para análise visual.
* **Jupyter Notebook:** Ambiente de desenvolvimento utilizado para documentação e execução do código de forma interativa.

---
