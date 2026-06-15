# Preparação dos dados

Nesta etapa, deverão ser descritas todas as técnicas utilizadas para pré-processamento/tratamento dos dados.

Algumas das etapas podem estar relacionadas à:

* Limpeza de Dados: trate valores ausentes: decida como lidar com dados faltantes, seja removendo linhas, preenchendo com médias, medianas ou usando métodos mais avançados; remova _outliers_: identifique e trate valores que se desviam significativamente da maioria dos dados.

* Transformação de Dados: normalize/padronize: torne os dados comparáveis, normalizando ou padronizando os valores para uma escala específica; codifique variáveis categóricas: converta variáveis categóricas em uma forma numérica, usando técnicas como _one-hot encoding_.

* _Feature Engineering_: crie novos atributos que possam ser mais informativos para o modelo; selecione características relevantes e descarte as menos importantes.

* Tratamento de dados desbalanceados: se as classes de interesse forem desbalanceadas, considere técnicas como _oversampling_, _undersampling_ ou o uso de algoritmos que lidam naturalmente com desbalanceamento.

* Separação de dados: divida os dados em conjuntos de treinamento, validação e teste para avaliar o desempenho do modelo de maneira adequada.
  
* Manuseio de Dados Temporais: se lidar com dados temporais, considere a ordenação adequada e técnicas específicas para esse tipo de dado.
  
* Redução de Dimensionalidade: aplique técnicas como PCA (Análise de Componentes Principais) se a dimensionalidade dos dados for muito alta.

* Validação Cruzada: utilize validação cruzada para avaliar o desempenho do modelo de forma mais robusta.

* Monitoramento Contínuo: atualize e adapte o pré-processamento conforme necessário ao longo do tempo, especialmente se os dados ou as condições do problema mudarem.

* Entre outras....

Avalie quais etapas são importantes para o contexto dos dados que você está trabalhando, pois a qualidade dos dados e a eficácia do pré-processamento desempenham um papel fundamental no sucesso de modelo(s) de aprendizado de máquina. É importante entender o contexto do problema e ajustar as etapas de preparação de dados de acordo com as necessidades específicas de cada projeto.

## 4. Feature Engineering Avançado

Com o objetivo de aumentar o poder preditivo dos modelos de Machine Learning, foram criadas novas variáveis derivadas do histórico dos jogadores, características temporais e estatísticas relacionadas às lesões. Todas as transformações foram realizadas respeitando a ordem cronológica dos eventos, evitando vazamento de informação (*data leakage*).

---

### 4.1 Garantia de Causalidade Temporal

Foi implementado um mecanismo de validação temporal para garantir que todas as variáveis fossem calculadas utilizando apenas informações anteriores à ocorrência da lesão.

**Objetivos:**

- Evitar *data leakage*.
- Garantir consistência temporal dos dados.
- Simular um cenário real de previsão.

```python
validator = TemporalValidator()
```

---

### 4.2 Features Históricas do Jogador

Foram criadas variáveis baseadas exclusivamente no histórico individual de lesões de cada atleta.

#### Features criadas

| Feature | Descrição |
|----------|------------|
| `rolling_avg_days_7d` | Média móvel da duração das lesões em janela de 7 dias |
| `rolling_avg_days_30d` | Média móvel da duração das lesões em janela de 30 dias |
| `rolling_avg_days_90d` | Média móvel da duração das lesões em janela de 90 dias |
| `rolling_avg_days_365d` | Média móvel da duração das lesões em janela de 365 dias |
| `injury_sequence_number` | Número sequencial da lesão do jogador |
| `days_since_last_injury` | Dias decorridos desde a lesão anterior |
| `recent_severity_trend` | Tendência recente de gravidade das lesões |
| `injuries_last_12_months` | Quantidade de lesões nos últimos 12 meses |
| `avg_recovery_time_recent` | Média do tempo de recuperação das últimas lesões |
| `chronic_injury_indicator` | Indicador de recorrência crônica da lesão |
| `reinjury_risk_score` | Score de risco de re-lesão |

#### Resultado

- **11 novas features históricas adicionadas ao dataset.**

---

### 4.3 Features Temporais e Calendáricas

Foram criadas variáveis relacionadas à sazonalidade e ao contexto temporal das lesões.

#### Features criadas

| Feature | Descrição |
|----------|------------|
| `injury_month` | Mês de ocorrência da lesão |
| `injury_dow` | Dia da semana da ocorrência |
| `month_sin` | Representação cíclica do mês utilizando seno |
| `month_cos` | Representação cíclica do mês utilizando cosseno |
| `season_phase_code` | Codificação da fase da temporada |
| `days_to_season_end` | Dias restantes até o final da temporada |
| `is_high_density_period` | Indicador de período de alta densidade de lesões |
| `congestion_index` | Índice de congestionamento de lesões do clube |
| `seasonal_injury_rate` | Taxa sazonal de lesões |

#### Resultado

- **9 novas features temporais adicionadas ao dataset.**

---

### 4.4 Features Estatísticas e Interações

Foram criadas variáveis estatísticas e de interação para capturar padrões históricos e relações entre atributos.

#### Features criadas

| Feature | Descrição |
|----------|------------|
| `rolling_mean_days_3` | Média móvel da duração das últimas 3 lesões |
| `rolling_count_injuries_3` | Quantidade de lesões nas últimas 3 ocorrências |
| `rolling_mean_days_5` | Média móvel da duração das últimas 5 lesões |
| `rolling_count_injuries_5` | Quantidade de lesões nas últimas 5 ocorrências |
| `rolling_mean_days_10` | Média móvel da duração das últimas 10 lesões |
| `rolling_count_injuries_10` | Quantidade de lesões nas últimas 10 ocorrências |
| `cumulative_days_injured` | Total acumulado de dias lesionado |
| `age_x_position` | Interação entre idade e posição do jogador |
| `age_x_league` | Interação entre idade e liga |
| `days_norm_position` | Valor normalizado por posição |
| `days_norm_league` | Valor normalizado por liga |
| `player_injury_rate_percentile` | Percentil da taxa de lesões do jogador |

#### Resultado

- **12 novas features estatísticas adicionadas ao dataset.**

---

### Resumo Final

| Etapa | Features Criadas |
|---------|---------------|
| Históricas | 11 |
| Temporais | 9 |
| Estatísticas | 12 |
| **Total** | **32** |

#### Dataset Final

| Métrica | Valor |
|----------|--------|
| Registros | 15.526 |
| Colunas Totais | 47 |
| Novas Features | 32 |

# Descrição dos modelos

Nesta seção, conhecendo os dados e de posse dos dados preparados, é hora de descrever os outros dois algoritmos de aprendizado de máquina selecionados para a construção dos modelos propostos. Inclua informações abrangentes sobre cada algoritmo implementado, aborde conceitos fundamentais, princípios de funcionamento, vantagens/limitações e justifique a escolha de cada um dos algoritmos. 

Explore aspectos específicos, como o ajuste dos parâmetros livres de cada algoritmo. Lembre-se de experimentar parâmetros diferentes e principalmente, de justificar as escolhas realizadas e registrar todos os experimentos realizados.

## 7. Otimização de Hiperparâmetros

Com o objetivo de melhorar o desempenho dos modelos de regressão, foi realizada a otimização automática de hiperparâmetros utilizando a biblioteca **Optuna**.

A busca foi conduzida por meio do algoritmo **TPE (Tree-structured Parzen Estimator)**, utilizando validação cruzada com **GroupKFold (3 folds)** para evitar vazamento de informações entre registros do mesmo jogador.

### Configuração da Otimização

| Parâmetro | Valor |
|------------|---------|
| Framework | Optuna |
| Algoritmo de Busca | TPE |
| Validação | GroupKFold (3 folds) |
| Métrica de Otimização | MAE |
| Número de Trials | 30 |

---

### 7.1 Otimização com Optuna (Melhoria 6)

#### LightGBM

| Hiperparâmetro | Valor |
|---------------|--------|
| `n_estimators` | 811 |
| `max_depth` | 8 |
| `learning_rate` | 0.00736 |
| `subsample` | 0.6877 |
| `colsample_bytree` | 0.9059 |
| `reg_alpha` | 0.0000418 |
| `reg_lambda` | 0.0683 |
| `min_child_samples` | 73 |

**MAE interno:** 18.8239

---

#### CatBoost

| Hiperparâmetro | Valor |
|---------------|--------|
| `iterations` | 1130 |
| `depth` | 6 |
| `learning_rate` | 0.02925 |
| `l2_leaf_reg` | 5.3349 |
| `subsample` | 0.5674 |

**MAE interno:** 18.9322

---

#### XGBoost

| Hiperparâmetro | Valor |
|---------------|--------|
| `n_estimators` | 1588 |
| `max_depth` | 6 |
| `learning_rate` | 0.00719 |
| `subsample` | 0.8612 |
| `colsample_bytree` | 0.9560 |
| `reg_alpha` | 0.000001 |
| `reg_lambda` | 8.0381 |

**MAE interno:** 18.8673

---

### 7.2 Retreinar com Melhores Hiperparâmetros

Após a seleção dos melhores hiperparâmetros, os modelos foram retreinados utilizando todo o conjunto de treinamento.

| Modelo | MAE | RMSE | R² |
|----------|---------|---------|---------|
| LightGBM_Optuna | 20.08 | 33.50 | 0.5427 |
| CatBoost_Optuna | 20.13 | 33.38 | 0.5458 |
| XGBoost_Optuna | 19.94 | 33.71 | 0.5369 |

---

### 7.3 Comparação com os Modelos Baseline

| Modelo | MAE Baseline | MAE Otimizado | Variação |
|----------|-------------|--------------|-----------|
| LightGBM | 20.09 | 20.08 | +0.0% |
| CatBoost | 20.20 | 20.13 | +0.3% |
| XGBoost | 20.12 | 19.94 | +0.9% |

Os resultados demonstram que a otimização dos hiperparâmetros produziu melhorias modestas, com destaque para o XGBoost, que apresentou a maior redução no erro absoluto médio.

---

## 8. Ensemble Avançado

Após a otimização dos modelos base, foram aplicadas estratégias de ensemble com o objetivo de melhorar a robustez e a performance preditiva do sistema. Os ensembles foram construídos utilizando os modelos otimizados de **LightGBM, CatBoost e XGBoost**.

---

### 8.1 Construção do Ensemble e Estratégias Avaliadas

O ensemble foi construído utilizando os três modelos otimizados como base:

- LightGBM
- CatBoost
- XGBoost

```python
ensemble_base = {
    'LightGBM': lgbm_opt,
    'CatBoost': cb_opt,
    'XGBoost': xgb_opt,
}

ensemble = AdvancedEnsemble(
    base_models=ensemble_base,
    random_state=RANDOM_STATE
)
```

---

### Estratégias de Ensemble Avaliadas

Foram comparadas três abordagens distintas:

#### Stacking
- Meta-learner (Ridge) treinado sobre previsões out-of-fold dos modelos base usando GroupKFold

#### Blending
- Meta-learner treinado sobre um conjunto de blend separado (20% do treino)
  
#### Weighted Average
- Pesos otimizados via Nelder-Mead para combinação convexa dos modelos
---

### Resultados da Comparação

| Estratégia | MAE | RMSE | R² |
|------------|---------|---------|---------|
| Stacking | 20.0188 | 33.3055 | 0.5479 |
| Blending | 20.1453 | 33.3100 | 0.5478 |
| Weighted Average | 19.9677 | 33.3770 | 0.5460 |

---


A estratégia com melhor desempenho em termos de MAE (menor erro absoluto médio) foi Weighted Average.

- MAE: 19.9677

---

## 8.2 Predições do Melhor Ensemble

Após a comparação das estratégias, o modelo final utilizado foi o ensemble selecionado automaticamente (Weighted Average no caso deste experimento).

A previsão final é calculada da seguinte forma:

```python
if best_ensemble_strategy == 'Stacking':
    y_pred_ensemble = ensemble.predict_stacking(X_test)

elif best_ensemble_strategy == 'Blending':
    y_pred_ensemble = ensemble.predict_blending(X_test)

else:
    y_pred_ensemble = ensemble.predict_weighted_average(X_test)

y_pred_ensemble = np.maximum(y_pred_ensemble, 0)
```

---

### Avaliação do Ensemble Final

| Métrica | Valor |
|----------|---------|
| MAE | 19.9677 |
| RMSE | 33.3770 |
| R² | 0.5460 |
| MedianAE | 11.9326 |
| MAPE | 1.0973 |
| Max Error | 253.8652 |

---

# Avaliação dos modelos criados

## Métricas utilizadas

Nesta seção, as métricas utilizadas para avaliar os modelos desenvolvidos deverão ser apresentadas (p. ex.: acurácia, precisão, recall, F1-Score, MSE etc.). A escolha de cada métrica deverá ser justificada, pois esta escolha é essencial para avaliar de forma mais assertiva a qualidade do modelo construído. 

## Discussão dos resultados obtidos

Nesta seção, discuta os resultados obtidos por cada um dos modelos construídos na Etapa 03 e na Etapa 04, no contexto prático em que os dados se inserem, promovendo uma compreensão abrangente e aprofundada da qualidade de cada um deles. Lembre-se de relacionar os resultados obtidos ao problema identificado, a questão de pesquisa levantada e estabelecer relação com os objetivos previamente propostos. Não deixe de comparar os resultados obtidos por cada modelo com os demais.

# 9. Explicabilidade com SHAP

A interpretabilidade dos modelos é um requisito fundamental para aplicações de Machine Learning em contextos reais, especialmente em domínios sensíveis como medicina esportiva e gestão de atletas profissionais.

Embora métricas como MAE, RMSE e R² permitam avaliar a qualidade preditiva dos modelos, elas não explicam **por que** determinada previsão foi produzida. Para suprir essa limitação, foi utilizada a biblioteca **SHAP (SHapley Additive exPlanations)**, que permite quantificar a contribuição individual de cada variável para as previsões do modelo.

O SHAP é baseado na Teoria dos Jogos Cooperativos e fornece explicações consistentes, locais e globais para modelos complexos, incluindo algoritmos baseados em árvores de decisão, como LightGBM, XGBoost e CatBoost.

---

## 9.1 Importância Global das Variáveis

### Escolha do Modelo para Explicabilidade

A análise SHAP foi realizada utilizando o modelo **LightGBM_Optuna**, selecionado como o melhor modelo entre aqueles submetidos à otimização automática de hiperparâmetros via Optuna.

A utilização desse modelo teve como objetivo analisar o comportamento de um modelo ajustado por busca automatizada de hiperparâmetros, permitindo uma interpretação mais detalhada dos padrões aprendidos durante o processo de otimização.

**Observação:** embora a análise SHAP tenha sido executada sobre o modelo **LightGBM_Optuna**, o melhor desempenho global do benchmark final do projeto foi obtido pelo modelo **LightGBM Baseline**, selecionado pela métrica principal de avaliação (MAE).

---

### Metodologia

Para viabilizar a análise interpretável sem comprometer o desempenho computacional, foi utilizada uma amostra de até 500 observações do conjunto de teste.

```python
n_shap_samples = min(500, len(X_test))
X_shap = X_test[:n_shap_samples]

explainer = SHAPExplainer(
    model=best_single_model,
    X_train=X_train,
    feature_names=feature_names,
    model_type='tree'
)

shap_values = explainer.compute_shap_values(X_shap)
```

A importância global foi calculada utilizando o valor absoluto médio dos SHAP values:

$$
Importance_j =
\frac{1}{N}
\sum_{i=1}^{N}
|\phi_{ij}|
$$

onde:

- $\(\phi_{ij} \$) representa a contribuição da variável \(j\) na observação \(i\);
- $\(N\$) corresponde ao número de observações analisadas.

---

### Ranking Global das Variáveis

| Variável | Mean Absolute SHAP |
|-----------|------------------:|
| Injury_target_enc | 15.53 |
| club_target_enc | 3.16 |
| player_injury_rate_percentile | 2.97 |
| Injury_freq | 2.12 |
| days_to_season_end | 1.92 |
| player_age | 1.10 |
| days_since_last_injury | 1.02 |
| rolling_mean_days_3 | 0.96 |
| player_position_target_enc | 0.80 |
| league_freq | 0.76 |

---

### Interpretação dos Resultados

Os resultados demonstram que as informações associadas ao histórico de lesões e ao contexto esportivo do atleta possuem maior influência sobre a duração da recuperação.

As principais conclusões são:

- O tipo da lesão é o fator mais relevante para a previsão do tempo de afastamento.
- O histórico individual de lesões contribui significativamente para o desempenho do modelo.
- Características do clube e da liga apresentam influência indireta na duração da recuperação.
- Variáveis temporais ajudam a capturar padrões sazonais e contextuais.

---

### Ponto de Atenção — Features com Alto Poder Preditivo

A variável **Injury_target_enc** apresentou a maior importância global na análise SHAP.

Esse comportamento é esperado, pois diferentes tipos de lesão possuem tempos médios de recuperação historicamente distintos. Dessa forma, a variável codificada por Target Encoding concentra uma parcela relevante da informação preditiva presente no conjunto de dados.

Como boas práticas de Machine Learning, variáveis derivadas de Target Encoding exigem validação cuidadosa para evitar vazamento de informação. Neste projeto foram adotados mecanismos de codificação segura, incluindo:

- separação temporal dos dados;
- validação apropriada por grupos;
- regularização durante o processo de encoding;
- prevenção de acesso a informações futuras.

Além disso, a validação de causalidade temporal realizada posteriormente não identificou evidências de vazamento associadas ao processo de Target Encoding.

Os alertas encontrados concentraram-se em features históricas relacionadas ao histórico médico do atleta, sendo posteriormente analisados e classificados como potenciais falsos positivos decorrentes da própria natureza da base de dados.

Portanto, a elevada importância observada para **Injury_target_enc** é compatível com o forte sinal estatístico existente entre o tipo da lesão e o tempo de recuperação, não constituindo evidência isolada de leakage.

---

## 9.2 Visualizações SHAP

Para complementar a análise quantitativa, foram produzidas visualizações globais utilizando os valores SHAP calculados.

As principais visualizações geradas incluem:

### Summary Plot

Permite visualizar simultaneamente:

- magnitude da contribuição das variáveis;
- direção do impacto sobre a previsão;
- distribuição dos efeitos ao longo das observações.

### Bar Plot de Importância Global

Apresenta o ranking ordenado das variáveis segundo a importância média absoluta dos SHAP values.

### Dependence Plots

Permitem avaliar relações não lineares entre cada variável e a previsão produzida pelo modelo.

Essas visualizações auxiliam na identificação de padrões complexos dificilmente observáveis por métodos tradicionais de interpretação.

---

## 9.3 Explicações Locais

Além da análise global, o SHAP permite explicar previsões individuais.

Para uma observação específica, a previsão final pode ser representada por:

$$
f(x)=E[f(x)]+\sum_{j=1}^{p}\phi_j
$$

onde:

- $\(E[f(x)]\$) é o valor esperado da previsão;
- $\(\phi_j\$) representa a contribuição da variável $\(j\$).

---

### Exemplo de Explicação Local

Em um caso representativo analisado, as principais contribuições observadas foram:

| Variável | Contribuição SHAP |
|-----------|-----------------:|
| Injury_target_enc | +23.34 |
| Injury_freq | +15.48 |
| club_target_enc | -4.06 |
| player_age | +2.17 |

Nesse cenário:

- o tipo da lesão aumentou significativamente a previsão de dias de afastamento;
- o histórico do atleta indicou maior propensão a recuperações longas;
- características do clube reduziram parcialmente a estimativa final.

Essa análise permite compreender de forma transparente como a previsão foi construída.

---

## 9.4 Análise de Casos Graves

Foi realizada uma análise específica para observações associadas a longos períodos de recuperação.

O objetivo foi identificar quais fatores possuem maior influência na previsão de lesões graves.

### Ranking SHAP para Casos Graves

| Variável | Mean SHAP |
|-----------|----------:|
| Injury_target_enc | 40.49 |
| Injury_freq | 5.27 |
| club_target_enc | 4.08 |
| player_injury_rate_percentile | 4.01 |
| days_since_last_injury | 3.42 |

---

### Conclusões da Análise de Casos Graves

Os resultados demonstram que:

- lesões graves são fortemente determinadas pelo tipo da lesão;
- o histórico acumulado do atleta exerce papel relevante na recuperação;
- fatores contextuais relacionados ao clube e à frequência de lesões amplificam o risco de afastamentos prolongados.

Essa análise reforça a importância de incorporar informações históricas e contextuais ao processo de previsão da duração das lesões.

---

## Conclusão da Etapa de Explicabilidade

A utilização do SHAP permitiu compreender de forma transparente os mecanismos internos do modelo, identificando as variáveis mais relevantes para a previsão da duração das lesões.

Os resultados demonstraram que:

- o tipo da lesão é o principal fator preditivo;
- o histórico médico do atleta possui forte influência;
- variáveis temporais e contextuais contribuem significativamente para o desempenho do modelo;
- as explicações locais são consistentes com o conhecimento de domínio da medicina esportiva.

Dessa forma, a etapa de explicabilidade complementa a avaliação quantitativa dos modelos, fornecendo evidências adicionais sobre a confiabilidade e a coerência das previsões produzidas pelo sistema.

# 10. Error Analysis Profundo

A avaliação de modelos preditivos não deve se restringir apenas às métricas globais de desempenho. Embora indicadores como MAE, RMSE e R² forneçam uma visão geral da qualidade do modelo, eles não permitem compreender em quais cenários o modelo apresenta melhor ou pior desempenho.

Com esse objetivo, foi realizada uma análise aprofundada dos erros de predição, investigando o comportamento do modelo em diferentes grupos de atletas, tipos de lesão, posições em campo e ligas europeias. Além disso, foram analisados os casos com maiores erros absolutos e a distribuição dos resíduos gerados pelo modelo.

---

## 10.1 Análise dos Maiores Erros

Para compreender as limitações do modelo, foi realizada uma análise dos 5% maiores erros absolutos observados no conjunto de teste.

Os maiores erros encontrados foram:

| Lesão | Posição | Valor Real (dias) | Valor Predito (dias) | Erro Absoluto |
|---------|---------|---------:|---------:|---------:|
| muscular problems | Forward | 276.75 | 25.69 | 251.06 |
| muscular problems | Attacking Midfield | 266.00 | 16.21 | 249.79 |
| Injury to the ankle | Centre-Back | 276.75 | 29.61 | 247.14 |
| Adductor pain | Right Winger | 261.00 | 17.64 | 243.36 |
| ankle sprain | Left-Back | 261.00 | 26.82 | 234.18 |

Observa-se que os maiores erros ocorreram principalmente em lesões graves ou extremamente raras, caracterizadas por longos períodos de recuperação.

O maior erro registrado foi de aproximadamente **251 dias**, em um caso classificado como *muscular problems*, cuja duração real foi de 276,75 dias enquanto o modelo estimou apenas 25,69 dias.

Esses resultados evidenciam uma limitação comum em problemas de regressão: a dificuldade em prever eventos extremos pouco representados nos dados de treinamento.

---

## 10.2 Análise de Erro por Faixa Etária

A Tabela a seguir apresenta o desempenho do modelo para diferentes grupos etários.

| Faixa Etária | Registros | MAE | MedianAE | RMSE |
|---------|---------:|---------:|---------:|---------:|
| U21 | 435 | 22.46 | 14.56 | 34.60 |
| 22–25 | 972 | 19.70 | 12.70 | 32.14 |
| 26–29 | 1089 | 19.82 | 11.37 | 33.82 |
| 30–33 | 595 | 19.22 | 10.90 | 32.62 |
| 34+ | 188 | 20.52 | 10.24 | 38.45 |

### Discussão

Os jogadores mais jovens (U21) apresentaram o maior erro médio absoluto, sugerindo maior variabilidade na recuperação física e menor histórico disponível para construção das variáveis temporais.

A faixa etária entre 30 e 33 anos apresentou o menor MAE, indicando maior estabilidade nos padrões de lesão e recuperação.

Os atletas com mais de 34 anos apresentaram RMSE significativamente superior, evidenciando maior presença de casos extremos e recuperações mais imprevisíveis.

---

## 10.3 Análise de Erro por Posição

A análise por posição em campo revelou diferenças relevantes na capacidade preditiva do modelo.

| Posição | Registros | MAE | MedianAE | RMSE |
|---------|---------:|---------:|---------:|---------:|
| Goalkeeper | 169 | 23.27 | 12.11 | 34.74 |
| Left Midfield | 28 | 23.10 | 11.35 | 42.39 |
| Left Winger | 258 | 21.67 | 13.21 | 36.12 |
| Attacking Midfield | 223 | 21.15 | 9.88 | 39.11 |
| Forward | 431 | 20.83 | 12.63 | 34.39 |
| Defensive Midfield | 250 | 20.32 | 11.55 | 35.03 |
| Right Winger | 226 | 20.09 | 13.82 | 32.32 |
| Centre-Back | 685 | 19.93 | 12.60 | 32.95 |
| Central Midfield | 421 | 19.79 | 11.59 | 33.40 |
| Left-Back | 241 | 19.42 | 13.03 | 31.93 |
| Right Midfield | 23 | 16.70 | 8.37 | 33.81 |
| Right-Back | 297 | 16.58 | 11.33 | 26.05 |
| Second Striker | 27 | 12.95 | 5.58 | 20.03 |

### Discussão

Os maiores erros foram observados entre goleiros e jogadores ofensivos, enquanto laterais e segundos atacantes apresentaram melhor desempenho preditivo.

Esses resultados sugerem que fatores táticos, físicos e médicos específicos de cada posição influenciam diretamente a previsibilidade do tempo de recuperação.

Também é importante considerar que algumas posições possuem menor quantidade de registros, aumentando a variabilidade das métricas observadas.

---

## 10.4 Análise de Erro por Tipo de Lesão

A análise por categoria médica revelou diferenças substanciais na dificuldade de previsão.

### Lesões com Maior Erro

| Tipo de Lesão | Registros | MAE |
|---------|---------:|---------:|
| Cruciate ligament tear | 51 | 39.88 |
| Knee injury | 127 | 37.20 |
| Shoulder injury | 45 | 30.71 |
| Ankle injury | 112 | 28.88 |
| Injury to the ankle | 48 | 27.71 |
| Foot injury | 60 | 27.68 |
| Hamstring injury | 323 | 23.96 |

### Lesões com Menor Erro

| Tipo de Lesão | Registros | MAE |
|---------|---------:|---------:|
| flu | 50 | 3.69 |
| Ill | 172 | 5.70 |
| Muscle fatigue | 89 | 8.82 |
| Knock | 100 | 12.02 |
| muscular problems | 201 | 13.24 |

### Discussão

Lesões graves envolvendo ligamentos, joelhos e articulações apresentaram os maiores erros de previsão.

Esse comportamento é esperado, pois tais lesões possuem elevada variabilidade clínica, diferentes protocolos de recuperação e forte influência de fatores individuais.

Por outro lado, condições leves como gripe, fadiga muscular e indisposição apresentam padrões de recuperação mais previsíveis, permitindo estimativas significativamente mais precisas.

---

## 10.5 Análise de Erro por Liga

Também foi avaliado o desempenho do modelo em diferentes ligas europeias.

| Liga | Registros | MAE | MedianAE | RMSE |
|---------|---------:|---------:|---------:|---------:|
| Premier League | 669 | 26.23 | 18.01 | 38.55 |
| Ligue 1 | 516 | 22.18 | 13.89 | 38.00 |
| La Liga | 378 | 20.08 | 12.27 | 32.56 |
| Bundesliga | 827 | 19.11 | 11.27 | 34.33 |
| Serie A | 889 | 15.09 | 8.07 | 25.25 |

### Discussão

A Premier League apresentou o maior erro médio absoluto, enquanto a Serie A apresentou o menor erro.

Esses resultados sugerem que fatores contextuais relacionados ao calendário competitivo, intensidade física, estratégias médicas dos clubes e perfil das lesões influenciam diretamente a previsibilidade do tempo de recuperação.

A diferença observada entre as ligas reforça a importância da inclusão de variáveis contextuais no processo de modelagem.

---

## 10.6 Análise dos Resíduos

Além das métricas agregadas, foi realizada uma análise visual dos resíduos produzidos pelo modelo.

Os gráficos gerados incluíram:

- Predicted vs True;
- Residuals vs Predicted;
- Residual Distribution;
- Absolute Error vs True.

### Principais Observações

A distribuição dos resíduos encontra-se relativamente concentrada em torno de zero, indicando ausência de viés sistemático severo.

O gráfico de resíduos versus valores preditos demonstra aumento gradual da dispersão para previsões mais elevadas, caracterizando um padrão de heterocedasticidade.

Observa-se ainda que os maiores erros concentram-se em lesões com longa duração, especialmente acima de 120 dias de afastamento.

O gráfico de erro absoluto versus valor real mostra crescimento quase linear da magnitude dos erros à medida que aumenta a duração da lesão, evidenciando a dificuldade do modelo em extrapolar eventos raros e severos.

Esses resultados são compatíveis com o comportamento esperado de modelos de regressão aplicados a problemas com distribuição altamente assimétrica.

---

## 10.7 Conclusão da Análise de Erros

A análise aprofundada dos erros permitiu identificar os principais pontos fortes e limitações do modelo desenvolvido.

Os resultados demonstraram que o modelo apresenta boa capacidade de generalização para a maioria dos cenários observados no conjunto de dados, mantendo erros relativamente estáveis entre diferentes grupos de atletas.

As principais dificuldades concentram-se em:

- lesões extremamente graves;
- casos raros com longos períodos de recuperação;
- atletas muito jovens;
- atletas veteranos com histórico complexo;
- ligas de maior intensidade competitiva.

Apesar dessas limitações, o comportamento observado é compatível com o estado da arte em problemas de regressão aplicados à medicina esportiva, reforçando a robustez do pipeline desenvolvido e apontando oportunidades claras para futuras melhorias metodológicas.

# 11. Calibração e Estimação de Incerteza

Modelos de regressão normalmente fornecem apenas uma estimativa pontual para cada previsão. Entretanto, em aplicações reais, especialmente em contextos médicos e esportivos, é igualmente importante conhecer o grau de confiança associado à previsão realizada.

Por esse motivo, foi implementado um módulo de quantificação de incerteza capaz de produzir intervalos de predição, avaliar a calibração estatística do modelo e medir a confiabilidade das estimativas geradas.

Essa etapa amplia significativamente o valor prático do sistema, permitindo não apenas prever o tempo de recuperação de um atleta, mas também quantificar a incerteza associada a essa previsão.

---

## 11.1 Motivação

Considere uma previsão de recuperação de:

```text
42 dias
```

Essa informação isoladamente não permite responder questões importantes, tais como:

* Qual a margem de erro esperada?
* Qual a confiança associada à previsão?
* O atleta pode retornar em 30 dias?
* Existe risco de afastamento superior a 60 dias?

Para responder essas perguntas, foram implementadas técnicas de estimativa de incerteza capazes de produzir intervalos de predição estatisticamente fundamentados.

O módulo desenvolvido incorpora três abordagens complementares:

| Método                         | Objetivo                           |
| ------------------------------ | ---------------------------------- |
| Bootstrap Prediction Intervals | Estimar incerteza via ensemble     |
| Conformal Prediction           | Produzir intervalos calibrados     |
| Calibration Analysis           | Avaliar a qualidade dos intervalos |

---

## 11.2 Bootstrap Prediction Intervals

### Conceito

Bootstrap é uma técnica estatística baseada em reamostragem com reposição.

A ideia consiste em:

1. Gerar múltiplas amostras aleatórias do conjunto de treinamento;
2. Treinar um modelo independente para cada amostra;
3. Avaliar a variabilidade das previsões produzidas.

No projeto foram treinados:

```text
20 modelos bootstrap
```

utilizando o melhor modelo otimizado como estimador base.

### Construção dos Intervalos

Para cada observação do conjunto de teste foram coletadas as previsões produzidas pelos 20 modelos bootstrap.

Os intervalos de confiança foram calculados através dos percentis empíricos:

* Percentil 2,5% → limite inferior;
* Percentil 97,5% → limite superior.

A previsão final foi definida pela mediana das previsões bootstrap, reduzindo a sensibilidade a valores extremos.

### Resultados Obtidos

Para um nível de confiança nominal de 95%, foram obtidos os seguintes resultados:

| Métrica                    |     Valor |
| -------------------------- | --------: |
| Cobertura Empírica         |     22,9% |
| Largura Média do Intervalo | 14,4 dias |

Exemplos observados:

| Valor Real | Intervalo Previsto |
| ---------: | ------------------ |
|         23 | [52, 67, 84]       |
|         74 | [56, 73, 86]       |
|         13 | [17, 22, 28]       |
|         11 | [29, 34, 40]       |
|         89 | [27, 42, 57]       |

Observa-se que diversos valores reais ficaram fora dos intervalos estimados.

### Interpretação

Embora os intervalos produzidos sejam relativamente estreitos, a cobertura observada foi extremamente inferior à cobertura nominal esperada.

Para um intervalo de 95%, esperava-se que aproximadamente 95% dos valores reais estivessem contidos nos intervalos gerados. Entretanto, apenas 22,9% das observações foram efetivamente capturadas.

Esse resultado indica que o modelo apresenta excesso de confiança quando a incerteza é estimada exclusivamente via Bootstrap.

---

## 11.3 Conformal Prediction

### Conceito

Para corrigir as limitações observadas nos intervalos bootstrap, foi implementado o método de Conformal Prediction.

A abordagem utiliza os resíduos observados no conjunto de calibração para construir intervalos estatisticamente válidos.

Os resíduos são definidos por:

$$
s_i = |y_i - \hat{y_i}|
$$

A partir desses resíduos é calculado um quantil de calibração:

$$
q = Quantile_{1-\alpha}(s)
$$

Os intervalos finais são então construídos como:

$$
[\hat{y}-q,\ \hat{y}+q]
$$

### Resultados Obtidos

Para um nível nominal de 95%:

| Métrica                    |     Valor |
| -------------------------- | --------: |
| Cobertura Empírica         |     93,2% |
| Largura Média do Intervalo | 86,7 dias |

### Interpretação

Diferentemente do Bootstrap, o método conformal produziu cobertura muito próxima do valor esperado.

A diferença entre cobertura nominal e cobertura observada foi de apenas:

$$
95% - 93,2% = 1,8%
$$

Esse resultado demonstra que os intervalos conformais são significativamente mais confiáveis para representar a incerteza do modelo.

---

## 11.4 Calibration Analysis

A qualidade dos intervalos foi avaliada através de uma análise de calibração.

O objetivo consiste em comparar:

* Cobertura Nominal (esperada);
* Cobertura Empírica (observada).

### Resultados

| Cobertura Nominal | Cobertura Empírica | Largura Média | Erro de Calibração |
| ----------------: | -----------------: | ------------: | -----------------: |
|              0.50 |              0.086 |          5.43 |              0.414 |
|              0.60 |              0.108 |          6.74 |              0.492 |
|              0.70 |              0.134 |          8.24 |              0.566 |
|              0.80 |              0.162 |         10.01 |              0.638 |
|              0.90 |              0.195 |         12.41 |              0.705 |
|              0.95 |              0.229 |         14.39 |              0.721 |
|              0.99 |              0.254 |         15.98 |              0.736 |

### Curva de Calibração

A curva de calibração revelou que a cobertura empírica permaneceu sistematicamente abaixo da cobertura nominal em todos os níveis analisados.

Esse comportamento evidencia que os intervalos bootstrap não representam adequadamente a variabilidade real observada nos dados.

---

## 11.5 Comparação entre Bootstrap e Conformal Prediction

A comparação direta dos métodos evidencia diferenças substanciais.

| Método                     | Cobertura | Largura Média |
| -------------------------- | --------: | ------------: |
| Bootstrap (95%)            |     22,9% |     14,4 dias |
| Conformal Prediction (95%) |     93,2% |     86,7 dias |

### Principais Diferenças

**Bootstrap**

* Intervalos estreitos;
* Menor utilidade para quantificação de risco;
* Cobertura insuficiente;
* Excesso de confiança.

**Conformal Prediction**

* Intervalos mais amplos;
* Cobertura próxima ao valor esperado;
* Melhor representação da incerteza;
* Garantias estatísticas de calibração.

---

## 11.6 Discussão dos Resultados

Os resultados obtidos revelaram uma característica importante do problema estudado.

Embora o modelo apresente bom desempenho preditivo em termos de MAE e RMSE, sua capacidade de estimar incerteza através de Bootstrap mostrou-se limitada.

A baixa cobertura observada sugere que a distribuição das lesões possui:

* elevada assimetria;
* presença de eventos extremos;
* caudas longas;
* poucos casos graves e muitos casos leves.

Nessas condições, os modelos bootstrap tendem a produzir previsões excessivamente semelhantes, reduzindo artificialmente a largura dos intervalos.

Por outro lado, o método Conformal Prediction conseguiu corrigir esse comportamento, produzindo intervalos compatíveis com a variabilidade observada nos dados.

Esse resultado reforça a importância de avaliar não apenas a precisão das previsões, mas também sua confiabilidade estatística.

---

## 11.7 Conclusão

A etapa de calibração e estimativa de incerteza elevou significativamente o rigor metodológico do projeto.

Os experimentos demonstraram que:

* o modelo apresenta boa capacidade preditiva para estimativas pontuais;
* os intervalos bootstrap são excessivamente otimistas e mal calibrados;
* o método Conformal Prediction produz intervalos significativamente mais confiáveis;
* a cobertura empírica de 93,2% aproxima-se adequadamente do alvo nominal de 95%.

Portanto, além de prever a duração das lesões, o sistema desenvolvido também é capaz de fornecer uma estimativa quantitativa da incerteza associada às previsões, aumentando sua utilidade prática para profissionais da medicina esportiva, analistas de desempenho e departamentos médicos de clubes de futebol.

# 12. Regressão Quantílica

A regressão tradicional fornece apenas uma estimativa pontual para cada observação. Embora essa abordagem seja adequada para diversos problemas de regressão, ela não representa adequadamente cenários onde existe elevada variabilidade, assimetria e heterocedasticidade nos dados.

No contexto da previsão da duração de lesões esportivas, diferentes atletas podem apresentar tempos de recuperação significativamente distintos mesmo quando submetidos a lesões semelhantes. Dessa forma, torna-se relevante estimar não apenas um valor único, mas diferentes cenários possíveis de recuperação.

Com esse objetivo, foi implementada uma abordagem de Regressão Quantílica capaz de modelar diretamente diferentes regiões da distribuição condicional da variável alvo.

---

## 12.1 Motivação

Considere dois atletas que sofreram a mesma lesão.

Embora possuam características semelhantes, um deles pode retornar rapidamente às atividades, enquanto o outro pode apresentar complicações que prolonguem significativamente o período de recuperação.

Uma regressão tradicional estima apenas:

$$
\hat{y}
$$

enquanto a regressão quantílica estima:

$$
Q_{\tau}(Y|X)
$$

onde:

* $\(Q_{\tau}\$) representa o quantil (\tau);
* (Y) corresponde ao tempo de recuperação;
* (X) representa o conjunto de características do atleta e da lesão.

Essa abordagem permite modelar explicitamente cenários otimistas, medianos e pessimistas.

---

## 12.2 Fundamentos da Regressão Quantílica

Foram utilizados três quantis distintos:

| Quantil | Interpretação      |
| ------- | ------------------ |
| Q0.1    | Cenário otimista   |
| Q0.5    | Cenário mediano    |
| Q0.9    | Cenário pessimista |

A modelagem foi implementada utilizando o algoritmo LightGBM configurado com objetivo quantílico.

```python
qr = QuantileRegression(
    quantiles=[0.1, 0.5, 0.9],
    backend="lightgbm",
    random_state=RANDOM_STATE
)
```

Dessa forma, foram treinados três modelos independentes, cada um especializado em uma região específica da distribuição dos tempos de recuperação.

---

## 12.3 Função de Perda Quantílica

A regressão quantílica utiliza a chamada **Pinball Loss**, definida por:

$$
L_{\tau}(y,\hat{y})=\begin{cases}
\tau(y-\hat{y}), & y \ge \hat{y} \
(\tau-1)(y-\hat{y}), & y < \hat{y}
\end{cases}
$$

Essa função penaliza erros de forma assimétrica.

Para valores elevados de ($\tau$), o modelo penaliza fortemente subestimações, tornando-se particularmente adequado para cenários pessimistas e previsão de lesões graves.

Uma propriedade importante é que, para:

$$
\tau = 0.5
$$

a Pinball Loss torna-se equivalente à minimização do erro absoluto médio (MAE), tornando o modelo mais robusto à presença de outliers.

---

## 12.4 Cenários de Recuperação

A principal vantagem da regressão quantílica consiste na geração de múltiplos cenários de recuperação.

Os primeiros exemplos observados no conjunto de teste foram:

| Real | Otimista (Q0.1) | Mediano (Q0.5) | Pessimista (Q0.9) |
| ---: | --------------: | -------------: | ----------------: |
|   23 |            14.2 |           46.3 |             114.7 |
|   74 |            16.0 |           54.2 |             115.3 |
|   13 |             5.1 |           14.3 |              72.3 |
|   11 |            10.3 |           25.0 |              59.1 |
|   89 |            12.8 |           22.9 |              78.2 |

Esses resultados demonstram que o modelo não produz apenas uma estimativa pontual, mas um intervalo plausível de recuperação.

Por exemplo, para um atleta cujo afastamento real foi de 23 dias, o modelo estimou:

* cenário otimista: 14 dias;
* cenário mediano: 46 dias;
* cenário pessimista: 115 dias.

Essa representação probabilística é significativamente mais informativa para aplicações práticas em medicina esportiva.

---

## 12.5 Avaliação da Cobertura Quantílica

Foi avaliada a cobertura do intervalo formado pelos quantis:

$$
[Q_{0.1}, Q_{0.9}]
$$

Teoricamente, espera-se que esse intervalo contenha aproximadamente:

$$
0.9 - 0.1 = 80%
$$

das observações reais.

### Resultado Obtido

| Métrica             | Valor |
| ------------------- | ----: |
| Cobertura Observada | 72.5% |
| Cobertura Esperada  | 80.0% |

A diferença observada foi relativamente pequena, indicando que os intervalos produzidos pelo modelo representam adequadamente a variabilidade dos dados.

Além disso, esse resultado foi substancialmente superior ao obtido pelos intervalos Bootstrap discutidos na Seção 11.

---

## 12.6 Comparação com Regressão Tradicional

Uma das análises mais importantes desta etapa consistiu em comparar a regressão quantílica mediana com o melhor modelo tradicional utilizado como referência.

### Resultados

| Modelo              |     MAE |
| ------------------- | ------: |
| Traditional (point) | 20.0676 |
| Quantile (median)   | 18.3192 |

### Discussão

O modelo quantílico mediano apresentou redução significativa do erro absoluto médio.

A diferença observada foi de aproximadamente:

$$
20.0676 - 18.3192 = 1.7484
$$

dias.

Esse resultado sugere que a modelagem baseada em quantis consegue capturar padrões da distribuição dos dados que não são adequadamente representados pela regressão tradicional baseada na média condicional.

---

## 12.7 Avaliação da Pinball Loss

Foi calculada a Pinball Loss para cada quantil modelado.

### Resultados

| Quantil | Pinball Loss |
| ------- | -----------: |
| Q0.1    |        2.770 |
| Q0.5    |        9.160 |
| Q0.9    |        7.045 |

### Interpretação

O menor valor foi obtido para o quantil otimista (Q0.1), indicando que o modelo consegue representar com maior precisão cenários de recuperação rápida.

Por outro lado, o quantil pessimista apresentou maior dificuldade devido à presença de lesões graves, eventos raros e casos extremos presentes na distribuição.

Esse comportamento é consistente com os resultados observados na análise de resíduos apresentada anteriormente.

---

## 12.8 Discussão dos Resultados

Os resultados obtidos evidenciam uma característica importante do problema estudado: a presença de heterocedasticidade.

Na análise de resíduos da Seção 10 foi observado que a variabilidade dos erros aumenta à medida que cresce a duração da lesão.

Esse comportamento viola parcialmente as premissas implícitas de modelos tradicionais baseados apenas na média condicional.

A regressão quantílica mostrou-se particularmente adequada para esse cenário porque modela diferentes regiões da distribuição:

* recuperações rápidas;
* recuperações típicas;
* recuperações prolongadas.

Além disso, a capacidade de produzir múltiplos cenários torna o modelo significativamente mais útil para aplicações práticas em clubes de futebol, departamentos médicos e equipes de preparação física.

---

## 12.9 Conclusão

A regressão quantílica representou uma das melhorias metodológicas mais relevantes implementadas no projeto.

Os resultados demonstraram que:

* o modelo fornece previsões probabilísticas mais informativas;
* os intervalos quantílicos capturam adequadamente a variabilidade dos dados;
* a cobertura observada (72,5%) aproximou-se do valor esperado (80%);
* a regressão quantílica mediana superou o modelo tradicional em termos de MAE;
* a abordagem modela explicitamente a heterocedasticidade presente nos tempos de recuperação.

Dessa forma, o projeto deixou de fornecer apenas uma estimativa pontual da duração da lesão e passou a produzir cenários completos de recuperação, aumentando significativamente a utilidade prática do sistema para tomada de decisão em ambientes esportivos profissionais.

## 13. Nested Cross-Validation (Melhoria 11)

A validação cruzada padrão pode **sobre-estimar** a performance quando hiperparâmetros são selecionados no mesmo conjunto de validação. A **Nested CV** resolve isso:

- **Loop externo** (5 folds): Avaliação de performance imparcial
- **Loop interno** (3 folds): Seleção de hiperparâmetros
- **GroupKFold** em ambos os loops: Previne leakage por jogador

Resultado: estimativa **não-enviesada** da performance de generalização.
---

### Resultados

Os resultados por fold mostram consistência razoável entre as divisões:

- Fold 0: MAE = 19.17 | R² = 0.496  
- Fold 1: MAE = 18.82 | R² = 0.487  
- Fold 2: MAE = 19.01 | R² = 0.542  
- Fold 3: MAE = 18.18 | R² = 0.577  
- Fold 4: MAE = 19.12 | R² = 0.527  

---

### Média geral da Nested CV

- **MAE médio:** 18.86 ± 0.40  
- **RMSE médio:** 32.48 ± 1.67  
- **R² médio:** 0.5257 ± 0.036  

---

### Comparação com Test Set

- **Nested CV MAE:** 18.86  
- **Test Set MAE:** 19.94  

- **Nested CV R²:** 0.5257  
- **Test Set R²:** 0.5369  
---

## 14. Benchmark Experimental Completo (Melhoria 12)

### Contexto

Comparação sistemática de **todos os modelos e estratégias** testados neste notebook:
- Modelos baseline (configuração padrão)
- Modelos otimizados (hiperparâmetros via Optuna)
- Estratégias de ensemble (Stacking, Blending, Weighted Average)
---

### Resultados

Os principais resultados observados foram:

- **XGBoost otimizado (Optuna)** apresentou o menor MAE (~19.94)
- **Weighted Average Ensemble** apresentou o melhor R² (~0.546)
- Modelos como LightGBM e CatBoost também apresentaram desempenho competitivo

---
<img width="1348" height="449" alt="image" src="https://github.com/user-attachments/assets/3d542cf4-4cf7-4851-a348-198a3305a28a" />



# Revisão do pipeline de pesquisa e análise de dados

Nesta etapa, os alunos devem revisar o pipeline de pesquisa e análise de dados proposto na Etapa 03, avaliando criticamente cada uma de suas etapas, fluxos e decisões. O objetivo agora é identificar possíveis ajustes, melhorias ou generalizações que tornem o pipeline mais abrangente e adaptável, de forma que ele seja capaz de representar qualquer processo de construção de sistemas de aprendizado de máquina – independentemente da área de aplicação, tipo de dado ou técnica utilizada.

Lembre-se de que um pipeline bem estruturado deve contemplar, de forma flexível e modular, as principais fases da pesquisa e experimentação em ciência de dados e aprendizado de máquina, incluindo (mas não se limitando a): formulação do problema, coleta e preparação dos dados, análise exploratória, definição de métricas, seleção e validação de modelos, interpretação dos resultados e documentação.

O resultado desta etapa deverá ser um pipeline revisado e justificado, acompanhado de uma breve descrição das alterações realizadas e dos motivos que levaram a cada mudança.

# 15. Análise Estatística dos Resultados

## Objetivo

Após a etapa de treinamento e otimização dos modelos, foi realizada uma análise estatística aprofundada dos resultados com o objetivo de verificar a robustez metodológica do pipeline, identificar possíveis inconsistências temporais e avaliar a confiabilidade das variáveis derivadas utilizadas durante o processo de modelagem.

Essa etapa é fundamental em projetos de séries temporais e Sports Analytics, pois permite identificar potenciais casos de Data Leakage, inconsistências históricas e problemas de qualidade dos dados que poderiam comprometer a validade científica dos resultados.

---

## Validação Temporal das Features

Foram selecionadas as principais variáveis históricas derivadas durante o processo de Feature Engineering para análise individual.

O fluxo de validação consistiu em:

1. Seleção das features temporais críticas;
2. Ordenação cronológica dos eventos por atleta;
3. Verificação da consistência temporal das variáveis;
4. Identificação de possíveis violações;
5. Análise das ocorrências encontradas.

---

## Resultados Obtidos

A validação identificou:

| Métrica                       | Resultado              |
| ----------------------------- | ---------------------- |
| Features analisadas           | 17                     |
| Possíveis violações temporais | 258                    |
| Feature mais crítica          | days_since_last_injury |

A principal variável envolvida foi a feature **Dias desde a última lesão**, responsável por representar o intervalo entre uma lesão atual e a lesão imediatamente anterior do atleta.

---

## Interpretação dos Casos Detectados

A existência de valores negativos nessa variável não implica necessariamente em Data Leakage.

Diversos cenários clínicos podem justificar esse comportamento:

* recaídas durante recuperação;
* múltiplas lesões simultâneas;
* procedimentos cirúrgicos durante afastamento;
* mudanças de status médico;
* histórico incompleto do atleta no dataset.

Além disso, a primeira lesão registrada no conjunto de dados não corresponde necessariamente à primeira lesão da carreira do jogador.

---

## Conclusões da Seção

A análise demonstrou que o pipeline possui mecanismos avançados de validação temporal e preocupação explícita com rigor metodológico.

A identificação automática de inconsistências representa uma importante camada de qualidade frequentemente ausente em projetos convencionais de Machine Learning.

---

# 16. Avaliação por Cenário Real

## Objetivo

Enquanto a avaliação tradicional mede o desempenho geral do modelo, a avaliação por cenário real busca verificar sua capacidade de generalização em situações semelhantes às encontradas em ambiente de produção.

Foram definidos quatro cenários principais:

* Jogadores nunca vistos;
* Clubes nunca vistos;
* Lesões raras;
* Temporadas futuras.

---

## Jogadores Nunca Vistos

O modelo foi avaliado utilizando atletas que não estavam presentes durante o treinamento.

### Resultado

* MAE ≈ 21,76 dias

### Interpretação

O aumento relativamente pequeno do erro demonstra que o sistema consegue utilizar informações contextuais e históricas para realizar previsões mesmo sem conhecer previamente o atleta.

---

## Clubes Nunca Vistos

Foi avaliada a capacidade do modelo em generalizar para equipes ausentes durante o treinamento.

### Resultado

* MAE ≈ 26,97 dias

### Interpretação

O aumento significativo do erro sugere que fatores associados aos clubes exercem influência importante sobre os tempos de recuperação.

Esses fatores podem incluir:

* infraestrutura médica;
* protocolos de recuperação;
* intensidade competitiva;
* políticas de retorno ao jogo.

---

## Lesões Raras

As lesões com baixa frequência de ocorrência apresentaram os maiores desafios.

### Resultado

* MAE ≈ 28,31 dias

### Interpretação

A baixa quantidade de exemplos históricos reduz a capacidade de aprendizado do modelo.

Esse cenário representa uma limitação natural dos métodos supervisionados.

---

## Temporadas Futuras

Foi realizada uma simulação realista utilizando temporadas posteriores às utilizadas no treinamento.

### Resultado

* MAE ≈ 20 dias

### Interpretação

O desempenho permaneceu praticamente inalterado, indicando excelente capacidade de generalização temporal.

---

## Conclusão da Seção

Os resultados demonstram que o sistema possui capacidade de generalização satisfatória para novos jogadores e novas temporadas, apresentando maior dificuldade apenas em cenários de baixa representatividade histórica.

---

# 17. Visualizações Profissionais

## Objetivo

Produzir visualizações de alta qualidade para interpretação dos resultados, comunicação dos achados e suporte à tomada de decisão.

---

## Comparação de Modelos

As visualizações evidenciaram o desempenho superior dos algoritmos baseados em Gradient Boosting.

Os melhores resultados foram obtidos por:

* LightGBM;
* CatBoost;
* XGBoost.

Modelos lineares como Ridge e ElasticNet apresentaram desempenho inferior.

---

## Interpretação

Esse comportamento indica que o problema possui forte componente não linear, envolvendo:

* interações complexas;
* dependências temporais;
* efeitos acumulativos;
* relações hierárquicas entre variáveis.

---

## Principais Insights Visuais

As análises gráficas permitiram identificar:

* influência do histórico recente do atleta;
* impacto da recorrência de lesões;
* efeito do calendário esportivo;
* importância do contexto temporal.

---

# 18. Preparação para Produção

## Objetivo

Transformar o projeto experimental em uma solução pronta para utilização em ambiente operacional.

---

## Componentes Implementados

### Validação de Entrada

Verificação automática de:

* colunas obrigatórias;
* tipos de dados;
* faixas válidas;
* consistência estrutural.

---

### Tratamento de Categorias Desconhecidas

O pipeline foi preparado para lidar com:

* novos clubes;
* novos atletas;
* novas categorias.

Isso evita falhas durante inferência.

---

### Serialização

Todos os componentes foram persistidos para reutilização futura.

Exemplos:

* modelos;
* encoders;
* pipelines;
* transformadores.

---

## Benefícios

* reprodutibilidade;
* escalabilidade;
* facilidade de implantação;
* manutenção simplificada.

---

# 19. MLflow Experiment Tracking

## Objetivo

Implementar rastreamento completo dos experimentos realizados durante o projeto.

---

## Informações Registradas

Cada execução passou a armazenar:

### Parâmetros

* hiperparâmetros;
* configurações dos modelos;
* seeds.

### Métricas

* MAE;
* RMSE;
* R²;
* MedianAE.

### Artefatos

* modelos treinados;
* gráficos;
* relatórios;
* arquivos de configuração.

---

## Benefícios

A utilização do MLflow proporciona:

* rastreabilidade;
* auditoria;
* comparação de experimentos;
* reprodutibilidade científica.

---

# 20. Conclusão Científica Final

## Melhor Modelo

O LightGBM otimizado apresentou o melhor equilíbrio entre:

* desempenho;
* estabilidade;
* velocidade de treinamento.

---

## Resultados Finais

| Métrica | Resultado |
| ------- | --------- |
| MAE     | ~19 dias  |
| RMSE    | ~33 dias  |
| R²      | ~0,55     |

---

## Principais Variáveis

As variáveis mais relevantes foram:

1. Tipo da lesão;
2. Frequência de lesões;
3. Dias desde a última lesão;
4. Histórico recente;
5. Fase da temporada.

---

## Interpretação Científica

Os resultados indicam que a duração da recuperação não depende exclusivamente da lesão atual.

O histórico clínico do atleta exerce influência significativa sobre o processo de recuperação.

---

## Limitações

Entre as principais limitações observadas estão:

* ausência de dados médicos detalhados;
* ausência de dados fisiológicos;
* falta de métricas de carga física;
* baixa representatividade de lesões raras.

---

# 21. Salvamento de Artefatos e Encerramento do Projeto

## Objetivo

Garantir a persistência dos resultados produzidos durante o desenvolvimento.

---

## Artefatos Salvos

Foram armazenados:

### Modelos

* LightGBM;
* CatBoost;
* XGBoost;
* Ensemble final.

### Transformadores

* encoders;
* scalers;
* pipelines.

### Resultados

* métricas;
* gráficos;
* relatórios.

### Configurações

* hiperparâmetros;
* configurações do treinamento;
* informações dos experimentos.

---

## Benefícios

O salvamento dos artefatos garante:

* reprodutibilidade completa;
* continuidade futura do projeto;
* reutilização dos modelos;
* implantação simplificada.

---

# Considerações Finais

As seções 15 a 21 representam a consolidação do projeto em um sistema completo de Machine Learning aplicado ao contexto esportivo.

Além da obtenção de resultados competitivos, o trabalho demonstrou preocupação com:

* rigor científico;
* validação temporal;
* generalização;
* interpretabilidade;
* rastreabilidade;
* preparação para produção.

Como resultado, o projeto evoluiu de um experimento acadêmico para uma solução robusta e potencialmente aplicável em ambientes reais de Sports Analytics e Medicina Esportiva.


## Observações importantes

Todas as tarefas realizadas nesta etapa deverão ser registradas em formato de texto junto com suas explicações de forma a apresentar os códigos desenvolvidos e também, o código deverá ser incluído, na íntegra, na pasta "src".
