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
# Trecho do código utilizado para converter e analisar a duração
df_injuries['duracao_dias'] = df_injuries['duracao_dias'].str.extract('(\d+)').astype(int)
estatisticas = df_injuries[['duracao_dias', 'jogos_perdidos', 'idade']].describe()
```

| Métrica | Duração (Dias) | Jogos Perdidos | Idade |
| :--- | :--- | :--- | :--- |
| **Média** | [Espaço para preencher] | [Espaço para preencher] | [Espaço para preencher] |
| **Mediana** | [Espaço para preencher] | [Espaço para preencher] | [Espaço para preencher] |
| **Desvio Padrão** | [Espaço para preencher] | [Espaço para preencher] | [Espaço para preencher] |

*Visualizações inseridas:*
* **Box Plots:** Foram gerados para identificar a severidade das lesões por posição do jogador e idade, permitindo a visualização clara de *outliers* (lesões de longa duração, como rupturas de ligamento cruzado que chegam a +200 dias).

<img width="1005" height="942" alt="output" src="https://github.com/user-attachments/assets/16b0c42e-a20f-4207-8496-d9f2266c59c9" />

<img width="1005" height="864" alt="output2" src="https://github.com/user-attachments/assets/eb0c85c2-7f29-4fa3-99a3-8fa032dad881" />


* **Histogramas:** [Espaço para descrever a distribuição da frequência das lesões].

---

## Descrição dos achados

A partir da EDA, os seguintes pontos foram identificados como cruciais para o futuro treinamento da IA:

1.  **Amplitude da Severidade:** As lesões variam de 1 dia a 1.013 dias de afastamento. A grande massa de dados concentra-se em lesões de curta duração, mas os *outliers* representam o maior impacto financeiro e técnico para os clubes.
2.  **Variabilidade por Posição:** [Espaço para preencher se atacantes se lesionam mais que zagueiros, conforme observado nos gráficos].
3.  **Correlações:**
    * **Forte:** Identificamos uma correlação positiva forte entre `duracao_dias` e `jogos_perdidos`.
    * **Moderada/Fraca:** [Espaço para descrever a relação entre idade e frequência de lesões].
4.  **Impacto Externo:** O dataset registra um volume significativo de afastamentos por "Corona virus" e "Quarantine" entre 2020 e 2022, o que deve ser tratado como um padrão atípico na engenharia de dados.

---

## Ferramentas utilizadas

O projeto foi desenvolvido utilizando o ecossistema de Ciência de Dados da linguagem **Python**, com as seguintes bibliotecas:

* **Pandas:** Manipulação de dados, tratamento de tipos e tradução de colunas.
* **NumPy:** Suporte para cálculos matemáticos e operações em arrays.
* **Matplotlib / Seaborn:** Geração de gráficos estatísticos, box plots e histogramas para análise visual.
* **Jupyter Notebook:** Ambiente de desenvolvimento utilizado para documentação e execução do código de forma interativa.

---

## Descrição dos achados

A partir da análise descrita e exploratória realizada, descreva todos os achados considerados relevantes para o contexto em que o trabalho se insere. Por exemplo: com relação à centralidade dos dados algo chamou a atenção? Foi possível identificar correlação entre os atributos? Que tipo de correlação (forte, fraca, moderada)? 

## Ferramentas utilizadas

Existem muitas ferramentas diferentes que podem ser utilizadas para fazer a análise dos dados. Nesta seção, descreva as ferramentas/tecnologias utilizadas e sua aplicação. Vale destacar que, preferencialmente, as análises deverão ser realizadas utilizando a linguagem de programação Python.


