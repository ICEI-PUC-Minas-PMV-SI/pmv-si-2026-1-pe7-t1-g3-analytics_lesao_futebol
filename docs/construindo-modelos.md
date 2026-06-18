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

### 7.1 Otimização com Optuna

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
| LightGBM_Optuna | 20.07 | 33.51 | 0.5424 |
| CatBoost_Optuna | 20.13 | 33.34 | 0.5471 |
| XGBoost_Optuna | 20.55 | 34.91 | 0.5369 |

---

### 7.3 Comparação com os Modelos Baseline

| Modelo | MAE Baseline | MAE Otimizado | Diferença |
|----------|---------:|---------:|---------:|
| LightGBM | 19.95 | 20.07 | -0.6% |
| CatBoost | 20.10 | 20.13 | -0.2% |
| XGBoost | 20.21 | 20.55 | -1.7% |

### Discussão dos Resultados

Durante a etapa de otimização, os modelos apresentaram excelentes resultados na validação cruzada interna, alcançando valores de MAE próximos de 18,8 dias.

Entretanto, após o retreinamento utilizando todo o conjunto de treinamento e a avaliação no conjunto de teste correspondente à temporada 24/25, observou-se que as versões otimizadas não superaram os modelos baseline.

O LightGBM Baseline permaneceu como o modelo com melhor desempenho entre todas as variantes avaliadas nesta etapa, apresentando MAE de 19,95 dias. As versões otimizadas de LightGBM, CatBoost e XGBoost obtiveram resultados ligeiramente inferiores.

Esse comportamento sugere que as configurações padrão dos algoritmos já estavam adequadamente ajustadas ao problema estudado e que os ganhos observados durante a validação cruzada não foram totalmente generalizados para dados não vistos.

Os resultados também reforçam uma das principais conclusões do projeto: os maiores ganhos de desempenho foram obtidos através da etapa de Feature Engineering e da construção de atributos temporais, enquanto a otimização de hiperparâmetros teve impacto limitado sobre a performance final dos modelos.

---

# 8. Ensemble Avançado

Após a otimização dos modelos base, foram aplicadas estratégias de ensemble com o objetivo de aumentar a robustez das previsões e investigar possíveis ganhos de desempenho através da combinação de diferentes algoritmos.

Foram utilizados como modelos base:

- LightGBM_Optuna
- CatBoost_Optuna
- XGBoost_Optuna

---

## 8.1 Construção do Ensemble e Estratégias Avaliadas

O ensemble foi construído utilizando os três modelos otimizados via Optuna.

```python
ensemble_base = {
    "LightGBM": lgbm_opt,
    "CatBoost": cb_opt,
    "XGBoost": xgb_opt,
}
```

Foram avaliadas três estratégias distintas:

### Stacking

Utiliza um meta-modelo (Ridge Regression) treinado sobre as previsões produzidas pelos modelos base utilizando GroupKFold.

### Blending

Treina um meta-modelo em um subconjunto separado do conjunto de treinamento.

### Weighted Average

Combina as previsões dos modelos utilizando pesos otimizados por meio do algoritmo Nelder-Mead.

---

## 8.2 Predições do Melhor Ensemble

Após a comparação das estratégias avaliadas, a estratégia selecionada foi:

**Weighted Average**

As métricas obtidas foram:

| Métrica | Valor |
|----------|---------:|
| MAE | 20.03 |
| RMSE | 33.33 |
| R² | 0.5472 |
| MedianAE | 11.95 |
| MAPE | 1.1063 |
| Max Error | 253.62 |

---

## 8.3 Discussão dos Resultados do Ensemble

O ensemble apresentou desempenho competitivo e bastante próximo dos melhores modelos individuais.

Entretanto, quando comparado ao benchmark completo do projeto, observou-se que o modelo LightGBM Baseline apresentou o menor erro absoluto médio (MAE), tornando-se o melhor modelo geral do experimento.

| Modelo | MAE |
|----------|---------:|
| LightGBM (Baseline) | 19.95 |
| Ensemble Weighted Average | 20.03 |
| LightGBM_Optuna | 20.07 |

A diferença entre o melhor modelo individual e o ensemble foi de aproximadamente 0,08 dias de MAE, valor extremamente pequeno na escala do problema.

Esse resultado sugere que os ganhos obtidos durante a etapa de Feature Engineering tiveram impacto mais significativo sobre o desempenho final do que a combinação de modelos ou a otimização de hiperparâmetros.

Apesar disso, o ensemble continua sendo uma alternativa robusta, pois reduz a dependência de um único algoritmo e tende a apresentar maior estabilidade em cenários distintos.

---

# 9. Explicabilidade com SHAP

A interpretabilidade dos modelos é um requisito fundamental para aplicações de Machine Learning em contextos reais, especialmente em domínios sensíveis como medicina esportiva e gestão de atletas profissionais.

Embora métricas como MAE, RMSE e R² permitam avaliar a qualidade preditiva dos modelos, elas não explicam por que determinada previsão foi produzida. Para suprir essa limitação, foi utilizada a biblioteca **SHAP (SHapley Additive exPlanations)**, que permite quantificar a contribuição individual de cada variável para as previsões do modelo.

O SHAP é baseado na Teoria dos Jogos Cooperativos e fornece explicações consistentes, locais e globais para modelos complexos, incluindo algoritmos baseados em árvores de decisão, como LightGBM, XGBoost e CatBoost.

---

## 9.1 Importância Global das Variáveis

### Escolha do Modelo para Explicabilidade

A análise SHAP foi realizada utilizando o modelo **LightGBM Baseline**, identificado como o melhor modelo global do benchmark experimental do projeto.

A escolha desse modelo garante que as explicações produzidas estejam diretamente associadas à solução final selecionada para o problema de previsão da duração de lesões.

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

* (\phi_{ij}) representa a contribuição da variável (j) na observação (i);
* (N) corresponde ao número de observações analisadas.

---

### Ranking Global das Variáveis

| Variável                      | Mean Absolute SHAP |
| ----------------------------- | -----------------: |
| Injury_target_enc             |              16.99 |
| club_target_enc               |               3.11 |
| player_injury_rate_percentile |               2.89 |
| days_to_season_end            |               1.91 |
| days_since_last_injury        |               1.09 |
| Injury_freq                   |               1.07 |
| player_age                    |               1.06 |
| rolling_mean_days_3           |               1.04 |
| league_freq                   |               0.85 |
| player_position_target_enc    |               0.78 |

---

### Interpretação dos Resultados

Os resultados demonstram que as informações associadas ao histórico de lesões e ao contexto esportivo do atleta possuem maior influência sobre a duração da recuperação.

As principais conclusões são:

* O tipo da lesão é o fator mais relevante para a previsão do tempo de afastamento.
* O histórico individual de lesões contribui significativamente para o desempenho do modelo.
* Características do clube e da liga apresentam influência indireta na duração da recuperação.
* Variáveis temporais ajudam a capturar padrões sazonais e contextuais.
* A proximidade do encerramento da temporada também exerce influência relevante nas previsões.

---

### Ponto de Atenção — Features com Alto Poder Preditivo

A variável **Injury_target_enc** apresentou a maior importância global na análise SHAP.

Esse comportamento é esperado, pois diferentes tipos de lesão possuem tempos médios de recuperação historicamente distintos. Dessa forma, a variável codificada por Target Encoding concentra uma parcela relevante da informação preditiva presente no conjunto de dados.

Como boas práticas de Machine Learning, variáveis derivadas de Target Encoding exigem validação cuidadosa para evitar vazamento de informação. Neste projeto foram adotados mecanismos de codificação segura, incluindo:

* separação temporal dos dados;
* validação apropriada por grupos;
* regularização durante o processo de encoding;
* prevenção de acesso a informações futuras.

Além disso, a validação de causalidade temporal realizada posteriormente não identificou evidências de vazamento associadas ao processo de Target Encoding.

Portanto, a elevada importância observada para **Injury_target_enc** é compatível com o forte sinal estatístico existente entre o tipo da lesão e o tempo de recuperação.

---

## 9.2 Visualizações SHAP

Para complementar a análise quantitativa, foram produzidas visualizações globais utilizando os valores SHAP calculados.

As principais visualizações geradas incluem:

### Summary Plot

Permite visualizar simultaneamente:

* magnitude da contribuição das variáveis;
* direção do impacto sobre a previsão;
* distribuição dos efeitos ao longo das observações.

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
f(x)=E[f(x)] + \sum_{j=1}^{p}\phi_j
$$

onde:

* (E[f(x)]) é o valor esperado da previsão;
* (\phi_j) representa a contribuição da variável (j).

---

### Exemplo de Explicação Local

Em um caso representativo analisado, as principais contribuições observadas foram:

| Variável                   | Contribuição SHAP |
| -------------------------- | ----------------: |
| Injury_target_enc          |            +20.84 |
| Injury_freq                |            +11.58 |
| club_target_enc            |             -4.50 |
| player_age                 |             +2.61 |
| rolling_mean_days_3        |             -1.54 |
| player_position_target_enc |             -1.51 |

Nesse cenário:

* o tipo da lesão aumentou significativamente a previsão de dias de afastamento;
* o histórico do atleta indicou maior propensão a recuperações prolongadas;
* características específicas do clube reduziram parcialmente a estimativa final;
* fatores relacionados à posição do atleta também contribuíram para o ajuste da previsão.

Essa análise permite compreender de forma transparente como a previsão foi construída.

---

## 9.4 Análise de Casos Graves

Foi realizada uma análise específica para observações associadas a longos períodos de recuperação (acima de 60 dias).

O objetivo foi identificar quais fatores possuem maior influência na previsão de lesões graves.

### Ranking SHAP para Casos Graves

| Variável                      | Mean SHAP |
| ----------------------------- | --------: |
| Injury_target_enc             |     44.63 |
| club_target_enc               |      4.04 |
| player_injury_rate_percentile |      3.78 |
| days_to_season_end            |      3.03 |
| Injury_freq                   |      2.39 |
| days_since_last_injury        |      2.08 |
| player_age                    |      1.50 |
| rolling_mean_days_3           |      1.45 |

---

### Conclusões da Análise de Casos Graves

Os resultados demonstram que:

* lesões graves são fortemente determinadas pelo tipo da lesão;
* o histórico acumulado do atleta exerce papel ainda mais relevante em recuperações prolongadas;
* fatores relacionados ao clube e ao contexto competitivo tornam-se mais importantes à medida que aumenta a severidade da lesão;
* o histórico recente de afastamentos contribui para identificar atletas com maior risco de longos períodos de recuperação.

Observa-se que a importância da variável **Injury_target_enc** aumenta substancialmente em comparação à análise global, indicando que o tipo da lesão se torna ainda mais determinante quando o afastamento é prolongado.

---

## Conclusão da Etapa de Explicabilidade

A utilização do SHAP permitiu compreender de forma transparente os mecanismos internos do modelo LightGBM Baseline, identificado como a melhor solução global do projeto.

Os resultados demonstraram que:

* o tipo da lesão é o principal fator preditivo da duração da recuperação;
* o histórico médico do atleta possui forte influência sobre as previsões;
* variáveis temporais e contextuais contribuem significativamente para o desempenho do modelo;
* lesões graves apresentam dependência ainda maior do histórico clínico e do tipo de lesão;
* as explicações locais são consistentes com o conhecimento de domínio da medicina esportiva.

Dessa forma, a etapa de explicabilidade complementa a avaliação quantitativa dos modelos, fornecendo evidências adicionais sobre a confiabilidade, coerência e interpretabilidade das previsões produzidas pelo sistema.

---

# 10. Análise Profunda de Erros

A avaliação de modelos preditivos não deve se limitar às métricas agregadas de desempenho. Mesmo modelos com bons resultados globais podem apresentar comportamentos distintos em diferentes subgrupos da população analisada.

Com o objetivo de compreender melhor as limitações do modelo selecionado, foi realizada uma análise aprofundada dos erros utilizando o modelo **LightGBM Baseline**, identificado como o melhor modelo do benchmark experimental.

Foram investigados:

* os maiores erros de previsão;
* o comportamento do erro por faixa etária;
* diferenças entre posições dos atletas;
* desempenho por tipo de lesão;
* comportamento do modelo em diferentes ligas europeias.

---

## 10.1 Maiores Erros de Predição

Inicialmente foram analisados os 5% maiores erros absolutos observados no conjunto de teste.

O limiar correspondente ao percentil 95 dos erros absolutos foi de aproximadamente **65,6 dias**, resultando em 164 observações classificadas como erros extremos.

### Exemplos dos Maiores Erros

| Tipo de Lesão       | Posição            |   Real | Previsto | Erro Absoluto |
| ------------------- | ------------------ | -----: | -------: | ------------: |
| muscular problems   | Forward            | 276.75 |    21.99 |        254.76 |
| muscular problems   | Attacking Midfield | 266.00 |    16.17 |        249.83 |
| Injury to the ankle | Centre-Back        | 276.75 |    31.98 |        244.77 |
| Adductor pain       | Right Winger       | 261.00 |    18.35 |        242.65 |
| ankle sprain        | Left-Back          | 261.00 |    25.25 |        235.75 |

### Discussão

Observa-se que os maiores erros ocorreram principalmente em lesões com tempos de recuperação extremamente elevados.

Esses casos representam eventos raros e altamente variáveis, dificultando a modelagem estatística. Em geral, o modelo tende a subestimar afastamentos excepcionalmente longos, fenômeno comum em problemas de regressão com distribuições assimétricas e presença de outliers.

---

## 10.2 Erro por Faixa Etária

Para investigar possíveis vieses relacionados à idade dos atletas, os erros foram segmentados em grupos etários.

| Faixa Etária | Casos |   MAE | MedianAE |  RMSE |
| ------------ | ----: | ----: | -------: | ----: |
| U21          |   435 | 22.31 |    14.55 | 34.52 |
| 22–25        |   972 | 19.53 |    12.67 | 31.80 |
| 26–29        |  1089 | 19.76 |    11.47 | 33.61 |
| 30–33        |   595 | 19.09 |    11.00 | 32.33 |
| 34+          |   188 | 20.53 |    11.04 | 38.45 |

### Discussão

Os atletas mais jovens (U21) apresentaram os maiores erros médios de previsão.

Esse comportamento pode estar associado à maior variabilidade fisiológica dos jogadores em início de carreira, bem como à menor quantidade de histórico disponível para construção das variáveis temporais.

Os atletas entre 30 e 33 anos apresentaram o menor MAE observado, indicando maior estabilidade das previsões nessa faixa etária.

---

## 10.3 Erro por Posição

A seguir foram avaliadas diferenças de desempenho entre posições de jogo.

| Posição            | Casos |   MAE |
| ------------------ | ----: | ----: |
| Left Midfield      |    28 | 23.55 |
| Goalkeeper         |   169 | 22.60 |
| Left Winger        |   258 | 21.59 |
| Attacking Midfield |   223 | 21.09 |
| Forward            |   431 | 20.53 |
| Defensive Midfield |   250 | 20.46 |
| Central Midfield   |   421 | 19.94 |
| Centre-Back        |   685 | 19.88 |
| Right Winger       |   226 | 19.82 |
| Left-Back          |   241 | 19.34 |
| Right Midfield     |    23 | 16.70 |
| Right-Back         |   297 | 16.25 |
| Second Striker     |    27 | 12.66 |

### Discussão

Os maiores erros ocorreram para jogadores das posições **Left Midfield** e **Goalkeeper**.

Entretanto, algumas dessas posições apresentam poucos exemplos na base, tornando as estimativas mais instáveis.

As posições defensivas apresentaram desempenho mais consistente, sugerindo padrões de lesão e recuperação mais previsíveis.

---

## 10.4 Erro por Tipo de Lesão

Uma das análises mais relevantes consiste na avaliação do desempenho por categoria de lesão.

### Top 20 Tipos de Lesão

| Tipo de Lesão          | Casos |   MAE |
| ---------------------- | ----: | ----: |
| Knee injury            |   127 | 37.41 |
| Cruciate ligament tear |    51 | 35.24 |
| Shoulder injury        |    45 | 30.12 |
| Ankle injury           |   112 | 28.57 |
| Foot injury            |    60 | 27.82 |
| Injury to the ankle    |    48 | 27.56 |
| Hamstring injury       |   323 | 24.18 |
| Calf injury            |    97 | 18.59 |
| Fitness                |   119 | 18.10 |
| Dead leg               |    68 | 17.95 |
| Adductor injury        |    48 | 17.59 |
| Thigh problems         |    87 | 16.95 |
| Muscle injury          |   205 | 16.87 |
| Adductor pain          |    77 | 16.77 |
| Knee problems          |    65 | 15.80 |
| muscular problems      |   201 | 13.16 |
| Knock                  |   100 | 12.07 |
| Muscle fatigue         |    89 |  8.25 |
| Ill                    |   172 |  5.41 |
| flu                    |    50 |  3.39 |

### Discussão

As lesões relacionadas ao joelho apresentaram os maiores erros médios.

Lesões como **Knee Injury**, **Cruciate Ligament Tear** e **Shoulder Injury** possuem processos de recuperação altamente variáveis, frequentemente dependentes de fatores clínicos não disponíveis na base de dados.

Por outro lado, condições menos severas e mais padronizadas, como **flu**, **illness** e **muscle fatigue**, apresentaram erros significativamente menores.

---

## 10.5 Erro por Liga

Por fim, foi realizada uma análise segmentada pelas principais ligas europeias.

| Liga           | Casos |   MAE |
| -------------- | ----: | ----: |
| Premier League |   669 | 25.69 |
| Ligue 1        |   516 | 22.14 |
| La Liga        |   378 | 20.01 |
| Bundesliga     |   827 | 19.19 |
| Serie A        |   889 | 15.05 |

### Discussão

A Premier League apresentou os maiores erros médios observados.

Uma possível explicação está relacionada à maior intensidade competitiva e ao ritmo físico característico da competição, fatores que podem aumentar a variabilidade dos tempos de recuperação.

Em contraste, a Serie A apresentou o menor MAE entre as ligas analisadas, indicando maior previsibilidade dos padrões de lesão observados nessa competição.

---

## 10.6 Conclusão da Análise de Erros

A análise detalhada dos erros permitiu identificar cenários nos quais o modelo apresenta maior dificuldade de generalização.

Os principais achados foram:

* lesões graves e de longa duração concentram os maiores erros de previsão;
* atletas muito jovens apresentam maior variabilidade nos tempos de recuperação;
* lesões relacionadas ao joelho representam os casos mais difíceis de modelar;
* diferenças entre ligas sugerem influência do contexto competitivo sobre a previsibilidade dos afastamentos;
* a maioria dos erros extremos está associada a eventos raros e pouco representados no conjunto de treinamento.

Esses resultados demonstram que o modelo apresenta desempenho consistente na maior parte dos casos, mas ainda enfrenta desafios em situações excepcionais caracterizadas por elevada complexidade clínica e grande variabilidade individual.

---

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

utilizando o modelo LightGBM Baseline como estimador base.

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
| Cobertura Empírica         |     44,3% |
| Largura Média do Intervalo | 25,7 dias |

### Exemplos Observados

| Valor Real | Intervalo Previsto |
| ---------: | ------------------ |
|         23 | [22, 47, 83]       |
|         74 | [43, 63, 111]      |
|         13 | [8, 20, 31]        |
|         11 | [23, 34, 46]       |
|         89 | [20, 43, 69]       |

Observa-se que diversos valores reais permanecem fora dos intervalos estimados, mesmo após o aumento da largura média dos intervalos.

### Interpretação

Embora os intervalos produzidos sejam relativamente estreitos, a cobertura observada foi substancialmente inferior à cobertura nominal esperada.

Para um intervalo de 95%, esperava-se que aproximadamente 95% dos valores reais estivessem contidos nos intervalos gerados. Entretanto, apenas 44,3% das observações foram efetivamente capturadas.

Esse resultado indica que o modelo apresenta excesso de confiança quando a incerteza é estimada exclusivamente via Bootstrap.

---

## 11.3 Conformal Prediction

### Conceito

Para corrigir as limitações observadas nos intervalos bootstrap, foi implementado o método de Conformal Prediction.

A abordagem utiliza os resíduos observados em um conjunto de calibração para construir intervalos estatisticamente válidos.

Os resíduos são definidos por:

A partir desses resíduos é calculado um quantil de calibração:

Os intervalos finais são então construídos como:

### Resultados Obtidos

Para um nível nominal de 95%, foram obtidos:

| Métrica                    |     Valor |
| -------------------------- | --------: |
| Cobertura Empírica         |     92,8% |
| Largura Média do Intervalo | 83,2 dias |

O quantil conformal calculado foi:

```text
q = 54,02 dias
```

### Interpretação

Diferentemente do Bootstrap, o método conformal produziu cobertura muito próxima do valor esperado.

A diferença entre cobertura nominal e cobertura observada foi de apenas:

```text
95,0% − 92,8% = 2,2%
```

Esse resultado demonstra que os intervalos conformais são significativamente mais confiáveis para representar a incerteza do modelo.

---

## 11.4 Calibration Analysis

A qualidade dos intervalos bootstrap foi avaliada através de uma análise de calibração.

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

A análise revelou que a cobertura empírica permaneceu sistematicamente abaixo da cobertura nominal em todos os níveis avaliados.

Esse comportamento evidencia que os intervalos bootstrap não representam adequadamente a variabilidade real observada nos dados.

---

## 11.5 Comparação entre Bootstrap e Conformal Prediction

A comparação direta dos métodos evidencia diferenças substanciais.

| Método                     | Cobertura | Largura Média |
| -------------------------- | --------: | ------------: |
| Bootstrap (95%)            |     44,3% |     25,7 dias |
| Conformal Prediction (95%) |     92,8% |     83,2 dias |

### Principais Diferenças

**Bootstrap**

* Intervalos relativamente estreitos;
* Menor capacidade de capturar eventos extremos;
* Cobertura insuficiente;
* Excesso de confiança.

**Conformal Prediction**

* Intervalos mais amplos;
* Cobertura próxima ao valor nominal;
* Melhor representação da incerteza;
* Garantias estatísticas de calibração.

---

## 11.6 Discussão dos Resultados

Os resultados obtidos revelaram uma característica importante do problema estudado.

Embora o modelo apresente bom desempenho preditivo em termos de MAE e RMSE, sua capacidade de estimar incerteza exclusivamente através de Bootstrap mostrou-se limitada.

A cobertura observada de apenas 44,3% sugere que a distribuição das lesões apresenta:

* elevada assimetria;
* presença de eventos extremos;
* caudas longas;
* grande variabilidade entre atletas e tipos de lesão.

Nessas condições, os modelos bootstrap tendem a produzir previsões excessivamente semelhantes, reduzindo artificialmente a largura dos intervalos.

Por outro lado, o método Conformal Prediction conseguiu corrigir esse comportamento, produzindo cobertura empírica de 92,8%, muito próxima do alvo nominal de 95%.

Esse resultado reforça a importância de avaliar não apenas a precisão das previsões, mas também sua confiabilidade estatística.

---

## 11.7 Conclusão

A etapa de calibração e estimativa de incerteza elevou significativamente o rigor metodológico do projeto.

Os experimentos demonstraram que:

* o modelo apresenta boa capacidade preditiva para estimativas pontuais;
* os intervalos bootstrap permanecem subcalibrados e excessivamente otimistas;
* o método Conformal Prediction produz intervalos significativamente mais confiáveis;
* a cobertura empírica de 92,8% aproxima-se adequadamente do alvo nominal de 95%.

Portanto, além de prever a duração das lesões, o sistema desenvolvido também é capaz de fornecer uma estimativa quantitativa da incerteza associada às previsões, aumentando sua utilidade prática para profissionais da medicina esportiva, analistas de desempenho e departamentos médicos de clubes de futebol.


---

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

* \(Q_{\tau}\) representa o quantil (\tau);
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
L_{\tau}(y,\hat{y})=
\begin{cases}
\tau(y-\hat{y}), & y \ge \hat{y} \\
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

---

# 13. Nested Cross-Validation

A avaliação de modelos de Machine Learning utilizando apenas uma única divisão treino-teste pode produzir estimativas excessivamente otimistas ou dependentes da partição utilizada.

Para aumentar o rigor metodológico da avaliação, foi implementado um procedimento de **Nested Cross-Validation**, considerado uma das abordagens mais robustas para estimativa do desempenho de modelos preditivos.

A técnica foi aplicada utilizando o modelo **LightGBM Baseline**, identificado como o melhor modelo do benchmark experimental.

---

## 13.1 Motivação

Em problemas envolvendo dados históricos de atletas, diferentes divisões dos dados podem produzir resultados distintos devido à variabilidade natural das lesões e dos perfis dos jogadores.

A Nested Cross-Validation permite reduzir esse risco ao avaliar o modelo em múltiplas partições independentes dos dados.

Além disso, essa abordagem fornece uma estimativa mais robusta da capacidade de generalização do modelo.

---

## 13.2 Configuração Experimental

Foi utilizado o seguinte protocolo:

| Parâmetro         | Valor             |
| ----------------- | ----------------- |
| Modelo            | LightGBM Baseline |
| Outer Folds       | 5                 |
| Inner Folds       | 3                 |
| Estratégia        | GroupKFold        |
| Métrica Principal | MAE               |

A utilização de GroupKFold garante que registros pertencentes ao mesmo jogador não sejam distribuídos simultaneamente entre treino e validação.

---

## 13.3 Resultados por Fold

| Fold |   MAE |  RMSE |     R² |
| ---- | ----: | ----: | -----: |
| 0    | 20.18 | 34.47 | 0.4649 |
| 1    | 19.95 | 32.90 | 0.4485 |
| 2    | 19.73 | 33.45 | 0.5254 |
| 3    | 19.13 | 31.12 | 0.5451 |
| 4    | 19.94 | 35.13 | 0.5044 |

---

## 13.4 Resultado Consolidado

A média dos resultados obtidos nos cinco folds foi:

| Métrica | Resultado       |
| ------- | --------------- |
| MAE     | 19.79 ± 0.40    |
| RMSE    | 33.41 ± 1.55    |
| R²      | 0.4977 ± 0.0405 |

O baixo desvio padrão observado para o MAE indica que o desempenho do modelo permaneceu estável ao longo das diferentes partições dos dados.

---

## 13.5 Comparação com o Holdout Final

Os resultados da Nested Cross-Validation foram comparados com aqueles obtidos no conjunto de teste final utilizado ao longo do projeto.

| Métrica |       Nested CV | Holdout |
| ------- | --------------: | ------: |
| MAE     |    19.79 ± 0.40 |   19.95 |
| R²      | 0.4977 ± 0.0405 |  0.5487 |

Observa-se que o MAE obtido pela validação aninhada foi extremamente próximo ao MAE observado no conjunto de teste final.

A diferença foi inferior a 0,2 dias, sugerindo forte consistência entre as diferentes estratégias de avaliação.

---

## 13.6 Discussão dos Resultados

Os resultados obtidos demonstram que o modelo apresenta boa capacidade de generalização.

A proximidade entre os valores de MAE obtidos na Nested Cross-Validation e no conjunto holdout sugere que o desempenho observado não depende de uma divisão específica dos dados.

Além disso, a baixa variabilidade entre os folds indica que o modelo se comporta de forma consistente para diferentes subconjuntos de jogadores.

Embora o valor médio de R² tenha sido ligeiramente inferior ao observado no conjunto de teste final, essa diferença é esperada devido à maior variabilidade dessa métrica em problemas de regressão com elevada dispersão da variável-alvo.

De forma geral, não foram observados indícios relevantes de overfitting.

---

## 13.7 Conclusão

A utilização da Nested Cross-Validation aumentou o rigor estatístico da avaliação realizada no projeto.

Os experimentos demonstraram que:

* o desempenho do modelo é consistente entre diferentes partições dos dados;
* o erro médio permanece estável ao longo dos folds;
* os resultados obtidos no conjunto holdout são compatíveis com os resultados da validação aninhada;
* não foram identificadas evidências significativas de sobreajuste.

Dessa forma, a Nested Cross-Validation fornece evidências adicionais de que o modelo desenvolvido possui capacidade real de generalização para novos casos de lesão.

---

# 14. Benchmark Experimental Completo

Após a execução de todas as etapas de modelagem, otimização, ensemble, explicabilidade e validação estatística, foi realizado um benchmark consolidado contendo todos os modelos avaliados ao longo do projeto.

O objetivo desta etapa foi comparar, em um único ambiente experimental, os modelos baseline, os modelos otimizados via Optuna e o melhor ensemble construído.

A métrica principal utilizada para ranqueamento foi o **MAE (Mean Absolute Error)**, por representar diretamente o erro médio em dias de recuperação.

---

## 14.1 Modelos Avaliados

Foram incluídos no benchmark:

### Modelos Baseline

* Random Forest
* Gradient Boosting
* Ridge Regression
* Elastic Net
* CatBoost
* XGBoost
* LightGBM

### Modelos Otimizados

* LightGBM_Optuna
* CatBoost_Optuna
* XGBoost_Optuna

### Ensemble

* Weighted Average (melhor estratégia identificada)

---

## 14.2 Resultados Consolidados

| Modelo                      |       MAE |      RMSE |         R² |  MedianAE |
| --------------------------- | --------: | --------: | ---------: | --------: |
| **[Baseline] LightGBM**     | **19.95** | **33.28** | **0.5487** | **11.99** |
| [Ensemble] WeightedAverage  |     20.03 |     33.33 |     0.5472 |     11.95 |
| [Optuna] LightGBM_Optuna    |     20.07 |     33.51 |     0.5424 |     12.10 |
| [Baseline] CatBoost         |     20.10 |     33.15 |     0.5522 |     12.36 |
| [Optuna] CatBoost_Optuna    |     20.13 |     33.34 |     0.5471 |     12.10 |
| [Baseline] XGBoost          |     20.21 |     33.63 |     0.5390 |     12.22 |
| [Baseline] RandomForest     |     20.26 |     33.44 |     0.5443 |     12.28 |
| [Optuna] XGBoost_Optuna     |     20.55 |     34.91 |     0.5033 |     11.47 |
| [Baseline] GradientBoosting |     20.81 |     34.55 |     0.5134 |     12.21 |
| [Baseline] Ridge            |     21.55 |     35.32 |     0.4915 |     13.31 |
| [Baseline] ElasticNet       |     22.00 |     34.87 |     0.5044 |     14.62 |

---

## 14.3 Ranking Final

A classificação final dos três melhores modelos foi:

| Posição | Modelo                    |   MAE |
| ------- | ------------------------- | ----: |
| 🥇 1º   | LightGBM Baseline         | 19.95 |
| 🥈 2º   | Ensemble Weighted Average | 20.03 |
| 🥉 3º   | LightGBM_Optuna           | 20.07 |

A diferença entre o primeiro e o segundo colocado foi de apenas:

```text
0,08 dias
```

enquanto a diferença entre o primeiro e o terceiro colocado foi de aproximadamente:

```text
0,11 dias
```

Essas diferenças são extremamente pequenas do ponto de vista prático.

---

## 14.4 Impacto da Otimização de Hiperparâmetros

Um dos resultados mais relevantes do benchmark foi observar que a otimização automática de hiperparâmetros não produziu melhorias significativas sobre os modelos baseline.

Por exemplo:

| Modelo   | Baseline | Optuna |
| -------- | -------: | -----: |
| LightGBM |    19.95 |  20.07 |
| CatBoost |    20.10 |  20.13 |
| XGBoost  |    20.21 |  20.55 |

Em todos os casos, o desempenho após a otimização permaneceu muito próximo ou ligeiramente inferior ao desempenho originalmente obtido pelos modelos baseline.

Esse comportamento sugere que os hiperparâmetros iniciais já se encontravam próximos de uma configuração adequada para o problema estudado.

---

## 14.5 Impacto do Ensemble

O ensemble baseado em Weighted Average apresentou desempenho competitivo:

| Modelo            |   MAE |
| ----------------- | ----: |
| LightGBM Baseline | 19.95 |
| Weighted Average  | 20.03 |

A diferença observada foi inferior a 0,1 dia.

Esse resultado indica que a combinação de modelos conseguiu manter desempenho semelhante ao melhor modelo individual, porém sem produzir ganhos relevantes de precisão.

Apesar disso, o ensemble continua sendo uma alternativa interessante por oferecer maior robustez ao combinar previsões de múltiplos algoritmos.

---

## 14.6 Discussão dos Resultados

O benchmark revelou um resultado particularmente interessante.

Embora tenham sido implementadas técnicas avançadas de:

* otimização via Optuna;
* ensembles;
* validação robusta;
* explicabilidade;
* calibração de incerteza;

o melhor desempenho foi obtido por um modelo LightGBM baseline.

Esse comportamento sugere que os maiores ganhos de desempenho do projeto não foram produzidos pela complexidade adicional dos algoritmos, mas sim pela qualidade das variáveis construídas durante a etapa de engenharia de atributos.

Em outras palavras, o sucesso do modelo parece estar mais associado à representação adequada do problema do que à utilização de técnicas de modelagem cada vez mais sofisticadas.

---

## 14.7 Conclusão

O benchmark experimental consolidou o **LightGBM Baseline** como o melhor modelo do projeto.

Os resultados demonstraram que:

* o LightGBM Baseline apresentou o menor MAE entre todos os modelos avaliados;
* os ensembles produziram desempenho competitivo, porém não superior;
* a otimização via Optuna não gerou ganhos relevantes de precisão;
* o Feature Engineering foi o principal responsável pelo desempenho alcançado;
* o modelo vencedor apresentou resultados consistentes nas etapas de validação, explicabilidade e análise de erros.

Dessa forma, o LightGBM Baseline foi selecionado como modelo final do sistema e posteriormente salvo como artefato principal para utilização em produção.

---

# 15. Análise Estatística dos Resultados

Após a avaliação dos modelos por meio das métricas tradicionais de regressão, foi realizada uma análise estatística complementar com o objetivo de quantificar a incerteza associada aos resultados obtidos e avaliar a robustez das diferenças observadas entre os modelos avaliados.

Embora métricas como MAE, RMSE e R² permitam comparar o desempenho médio dos modelos, elas não fornecem informações sobre a variabilidade dessas estimativas. Por esse motivo, foram utilizados intervalos de confiança obtidos por Bootstrap e testes estatísticos de comparação entre modelos.

---

## 15.1 Intervalos de Confiança por Bootstrap

Para estimar a variabilidade das métricas de desempenho, foi aplicada a técnica de Bootstrap com 1000 reamostragens sobre o conjunto de teste.

O procedimento consiste em:

1. Reamostrar aleatoriamente o conjunto de teste com reposição;
2. Recalcular a métrica de interesse para cada amostra bootstrap;
3. Construir a distribuição empírica da métrica;
4. Extrair os percentis correspondentes ao intervalo de confiança desejado.

Essa abordagem permite estimar a estabilidade dos resultados sem assumir distribuições paramétricas específicas.

---

### Intervalo de Confiança para o MAE

Utilizando o modelo LightGBM Baseline, identificado como o melhor modelo do benchmark experimental, foi obtido o seguinte intervalo de confiança para o erro absoluto médio:

| Métrica | Estimativa |          IC 95% |
| ------- | ---------: | --------------: |
| MAE     |      19.95 | [19.07 ; 20.86] |

O intervalo relativamente estreito indica que o erro médio do modelo permanece estável sob diferentes reamostragens dos dados.

---

### Intervalo de Confiança para o R²

Também foi calculado o intervalo de confiança para o coeficiente de determinação.

| Métrica | Estimativa |            IC 95% |
| ------- | ---------: | ----------------: |
| R²      |     0.5487 | [0.4993 ; 0.5976] |

Os resultados demonstram que o modelo mantém capacidade explicativa consistente mesmo quando submetido a diferentes amostras bootstrap.

---

## 15.2 Comparação Estatística entre Modelos

Além da avaliação individual do melhor modelo, foram calculados intervalos de confiança para todos os modelos baseline avaliados.

### Intervalos de Confiança do MAE

| Modelo           |   MAE | Limite Inferior | Limite Superior |
| ---------------- | ----: | --------------: | --------------: |
| LightGBM         | 19.95 |           19.07 |           20.86 |
| CatBoost         | 20.10 |           19.23 |           20.97 |
| XGBoost          | 20.21 |           19.33 |           21.09 |
| RandomForest     | 20.26 |           19.36 |           21.15 |
| GradientBoosting | 20.81 |           19.88 |           21.73 |
| Ridge            | 21.55 |           20.64 |           22.49 |
| ElasticNet       | 22.00 |           21.12 |           22.89 |

---

### Interpretação

Observa-se uma forte sobreposição entre os intervalos de confiança dos principais modelos baseados em árvores.

Por exemplo:

* LightGBM: [19.07 ; 20.86]
* CatBoost: [19.23 ; 20.97]
* XGBoost: [19.33 ; 21.09]

Essa sobreposição sugere que as diferenças observadas entre os melhores modelos são relativamente pequenas quando considerada a variabilidade amostral dos dados.

Embora o LightGBM tenha obtido o menor MAE observado no benchmark, a vantagem estatística sobre os demais modelos é modesta.

---

## 15.3 Teste Estatístico entre Modelos

Como demonstração adicional de inferência estatística, foi realizado um teste t pareado entre dois modelos baseline avaliados durante os experimentos.

### Resultados

| Estatística              |  Valor |
| ------------------------ | -----: |
| t-statistic              | -3.303 |
| p-value                  | 0.0010 |
| Significativo (α = 0.05) |    Sim |

O teste indicou diferença estatisticamente significativa entre os modelos comparados.

Entretanto, esse resultado deve ser interpretado com cautela, pois o experimento foi aplicado apenas a um par específico de modelos e não representa uma comparação exaustiva entre todos os participantes do benchmark.

---

## 15.4 Discussão dos Resultados

Os resultados obtidos fornecem evidências adicionais sobre a robustez do modelo selecionado.

A análise bootstrap demonstrou que:

* o MAE apresenta baixa variabilidade;
* o desempenho do LightGBM permanece estável sob diferentes reamostragens;
* os intervalos de confiança são compatíveis com os resultados observados no benchmark e na Nested Cross-Validation.

Além disso, a forte sobreposição dos intervalos de confiança dos melhores modelos sugere que parte das diferenças observadas no ranking pode estar associada à variabilidade natural dos dados.

Esse resultado reforça uma conclusão importante do projeto: os ganhos obtidos não decorreram exclusivamente da escolha do algoritmo, mas principalmente da qualidade das variáveis construídas durante a etapa de Feature Engineering.

---

## 15.5 Conclusão

A análise estatística complementou a avaliação tradicional dos modelos, fornecendo uma visão mais abrangente sobre a confiabilidade dos resultados.

Os experimentos demonstraram que:

* o LightGBM apresentou o menor MAE observado no benchmark;
* os intervalos de confiança confirmam a estabilidade do modelo;
* os melhores algoritmos apresentaram desempenhos muito próximos entre si;
* a variabilidade observada é relativamente pequena quando comparada à magnitude dos erros médios;
* os resultados obtidos são consistentes com as análises anteriores de benchmark, Nested Cross-Validation e explicabilidade.

Dessa forma, as evidências estatísticas reforçam a escolha do LightGBM Baseline como modelo final do projeto e aumentam a confiança na capacidade de generalização das previsões produzidas.

---

# 16. Avaliação por Cenário Real

## Objetivo

Embora métricas agregadas como MAE, RMSE e R² sejam fundamentais para avaliar o desempenho global de um modelo, elas não necessariamente refletem seu comportamento em situações reais encontradas durante a operação do sistema.

Por esse motivo, foi realizada uma avaliação baseada em cenários práticos de utilização, com o objetivo de investigar a capacidade de generalização do modelo em condições mais desafiadoras do que aquelas observadas durante o treinamento.

Foram considerados quatro cenários representativos:

* Jogadores nunca vistos durante o treinamento;
* Clubes nunca vistos durante o treinamento;
* Lesões raras com baixa frequência histórica;
* Dados de temporadas futuras.

Essa abordagem permite avaliar não apenas a precisão do modelo, mas também sua robustez diante de situações comuns em ambientes reais de análise esportiva.

---

## 16.1 Jogadores Nunca Vistos

Neste cenário foram avaliados apenas atletas que não possuíam registros no conjunto de treinamento.

O objetivo foi verificar se o modelo consegue realizar previsões adequadas mesmo sem histórico prévio do jogador.

### Resultados

| Métrica  |  Valor |
| -------- | -----: |
| MAE      |  21.67 |
| RMSE     |  35.04 |
| R²       | 0.6218 |
| MedianAE |  13.71 |

### Discussão

Comparado ao desempenho geral do modelo (MAE = 19.95), observa-se um aumento moderado do erro.

Apesar da ausência de histórico individual, o modelo manteve desempenho satisfatório, indicando que as variáveis contextuais relacionadas à lesão, posição, idade, clube e liga conseguem compensar parcialmente a falta de informações específicas do atleta.

Esse resultado demonstra boa capacidade de generalização para novos jogadores, cenário bastante comum em aplicações reais envolvendo transferências, promoções de atletas das categorias de base e novas contratações.

---

## 16.2 Clubes Nunca Vistos

Neste experimento foram selecionados clubes que não estavam presentes durante o treinamento.

O objetivo foi avaliar a capacidade de generalização do modelo para equipes inéditas.

### Resultados

| Métrica  |  Valor |
| -------- | -----: |
| MAE      |  26.47 |
| RMSE     |  45.86 |
| R²       | 0.4688 |
| MedianAE |  14.87 |

### Discussão

Esse cenário apresentou uma deterioração mais significativa do desempenho.

O aumento do MAE sugere que fatores associados ao clube exercem influência importante na duração das lesões.

Entre esses fatores podem estar:

* qualidade da estrutura médica;
* protocolos de recuperação;
* intensidade competitiva;
* carga de jogos;
* perfil físico dos atletas;
* estratégias de retorno ao jogo.

Os resultados indicam que informações relacionadas ao clube possuem valor preditivo relevante e que a expansão da base de treinamento para incluir um maior número de equipes pode contribuir para aumentar a capacidade de generalização do sistema.

---

## 16.3 Lesões Raras

As lesões raras foram definidas como aquelas com menos de cinco ocorrências históricas na base de dados.

### Resultados

| Métrica  |  Valor |
| -------- | -----: |
| MAE      |  28.43 |
| RMSE     |  42.71 |
| R²       | 0.2814 |
| MedianAE |  17.80 |

### Discussão

Este foi o cenário mais desafiador de todo o experimento.

O aumento expressivo do erro era esperado, uma vez que algoritmos supervisionados dependem diretamente da disponibilidade de exemplos representativos durante o treinamento.

Quando determinadas lesões possuem poucas observações históricas, o modelo dispõe de informações insuficientes para aprender adequadamente seus padrões de recuperação.

Esse resultado evidencia uma limitação inerente ao problema e não necessariamente uma deficiência do algoritmo utilizado.

Uma possível estratégia futura para mitigar esse efeito consiste na ampliação da base histórica ou na utilização de abordagens hierárquicas capazes de agrupar lesões clinicamente semelhantes.

---

## 16.4 Temporada Futura

O cenário temporal foi construído utilizando registros da temporada 2024/2025, mantidos totalmente separados durante o treinamento.

Essa configuração simula uma situação real de implantação do sistema, na qual o modelo é utilizado para realizar previsões sobre eventos futuros.

### Resultados

| Métrica  |  Valor |
| -------- | -----: |
| MAE      |  19.95 |
| RMSE     |  33.28 |
| R²       | 0.5487 |
| MedianAE |  11.99 |

### Discussão

O desempenho observado foi praticamente idêntico ao obtido no conjunto de teste geral.

Esse resultado representa uma importante evidência de validade temporal, indicando que o modelo foi capaz de manter sua capacidade preditiva mesmo quando aplicado a dados efetivamente futuros.

A estabilidade observada sugere que as variáveis construídas durante a etapa de Feature Engineering capturaram padrões relativamente consistentes ao longo do tempo, reduzindo o risco de sobreajuste a temporadas específicas.

---

## 16.5 Comparação Consolidada dos Cenários

A Tabela a seguir resume os resultados obtidos em todos os cenários avaliados.

| Cenário          |   MAE |  RMSE |     R² |
| ---------------- | ----: | ----: | -----: |
| Geral (Test Set) | 19.95 | 33.28 | 0.5487 |
| Jogador Novo     | 21.67 | 35.04 | 0.6218 |
| Clube Novo       | 26.47 | 45.86 | 0.4688 |
| Lesão Rara       | 28.43 | 42.71 | 0.2814 |
| Temporada 24/25  | 19.95 | 33.28 | 0.5487 |

---

## 16.6 Conclusão

A avaliação por cenários reais demonstrou que o modelo LightGBM mantém desempenho consistente em situações próximas às encontradas em produção.

Os principais achados desta etapa foram:

* boa capacidade de generalização para atletas nunca vistos;
* forte estabilidade temporal ao prever dados da temporada futura;
* maior sensibilidade à ausência de informações relacionadas aos clubes;
* dificuldade adicional na modelagem de lesões raras devido à baixa representatividade histórica.

De forma geral, os resultados indicam que o modelo possui potencial para utilização prática em ambientes de Sports Analytics, apresentando desempenho robusto mesmo quando submetido a cenários mais desafiadores do que aqueles utilizados durante o treinamento.

---

# 17. Visualizações Profissionais

## Objetivo

Além da avaliação quantitativa dos modelos, foram produzidas visualizações analíticas com o objetivo de facilitar a interpretação dos resultados, identificar padrões relevantes nos dados e comunicar os principais achados do projeto de forma clara e intuitiva.

As visualizações geradas contemplam tanto aspectos relacionados ao desempenho dos modelos quanto análises de comportamento das variáveis mais relevantes para a previsão da duração das lesões.

Essa etapa contribui para aumentar a transparência do processo de modelagem e fornecer evidências visuais que complementam as métricas estatísticas apresentadas nas seções anteriores.

---

## 17.1 Learning Curves

As curvas de aprendizado foram utilizadas para analisar o comportamento do modelo LightGBM à medida que o volume de dados de treinamento aumenta.

A Figura correspondente apresenta a evolução simultânea do erro de treinamento (Train MAE) e do erro de validação (Validation MAE) para diferentes tamanhos do conjunto de treinamento.

### Resultados Observados

Os resultados indicaram que:

* o erro de treinamento aumenta gradualmente conforme novos dados são incorporados;
* o erro de validação diminui de forma consistente;
* a distância entre as curvas reduz-se progressivamente com o aumento da amostra.

Nos primeiros experimentos, utilizando aproximadamente 800 observações, o modelo apresentou:

* Train MAE ≈ 2,5 dias;
* Validation MAE ≈ 24,2 dias.

Com a utilização de todo o conjunto de treinamento, os resultados convergiram para aproximadamente:

* Train MAE ≈ 9,6 dias;
* Validation MAE ≈ 19,9 dias.

### Interpretação

O comportamento observado é característico de modelos que inicialmente apresentam tendência ao sobreajuste quando treinados com poucos dados, mas que passam a generalizar melhor à medida que novas observações são incorporadas.

A redução gradual do erro de validação sugere que o modelo continua se beneficiando do aumento da base histórica, indicando potencial para ganhos adicionais de desempenho caso novos dados sejam disponibilizados futuramente.

Além disso, a ausência de crescimento do erro de validação nas maiores amostras sugere que não há evidências relevantes de underfitting ou degradação da capacidade preditiva.

---

## 17.2 Comparação da Importância das Variáveis

Foi realizada uma análise comparativa da importância das variáveis utilizando os três principais modelos baseados em Gradient Boosting avaliados no benchmark:

* LightGBM;
* CatBoost;
* XGBoost.

O objetivo foi verificar se diferentes algoritmos convergem para um conjunto semelhante de fatores explicativos.

### Resultados Observados

Os três modelos identificaram a variável **Injury_target_enc** como o fator mais relevante para a previsão da duração das lesões.

Além disso, observou-se elevada consistência entre os algoritmos na identificação das seguintes variáveis como importantes:

* Injury_target_enc;
* Injury_freq;
* club_target_enc;
* days_since_last_injury;
* player_injury_rate_percentile;
* cumulative_days_injured;
* days_to_season_end.

### Interpretação

A convergência observada entre diferentes algoritmos aumenta a confiabilidade das conclusões obtidas, indicando que os principais padrões identificados não dependem exclusivamente de um modelo específico.

Os resultados reforçam as evidências obtidas na análise SHAP, demonstrando que:

* o tipo da lesão é o principal determinante da duração da recuperação;
* o histórico médico do atleta possui forte capacidade preditiva;
* características temporais e contextuais contribuem significativamente para o desempenho do modelo;
* informações relacionadas ao clube e ao calendário esportivo influenciam o tempo de afastamento.

---

## Principais Insights Visuais

A análise conjunta das visualizações produzidas ao longo do projeto permitiu identificar diversos padrões relevantes para o problema estudado.

Entre os principais achados destacam-se:

* forte influência do histórico de lesões na duração da recuperação;
* impacto significativo da frequência de lesões anteriores;
* relevância do tipo da lesão como principal fator preditivo;
* contribuição das características do clube e da liga;
* influência de fatores temporais relacionados ao calendário esportivo;
* existência de comportamentos distintos entre lesões leves e lesões graves.

Esses resultados demonstram que a duração das lesões não depende de um único fator isolado, mas sim da interação entre características médicas, históricas, temporais e contextuais.

---

## Conclusão da Seção

As visualizações produzidas complementam as análises quantitativas apresentadas ao longo do projeto, fornecendo evidências visuais sobre o comportamento dos modelos e das variáveis explicativas.

As curvas de aprendizado demonstraram que o modelo LightGBM apresenta boa capacidade de generalização e potencial para melhorias adicionais com a incorporação de novos dados. Já a comparação das importâncias das variáveis confirmou a relevância do histórico médico do atleta e do tipo da lesão como os principais determinantes da duração do afastamento.

Em conjunto, essas análises fortalecem a interpretação dos resultados e aumentam a confiabilidade das conclusões obtidas pelo sistema desenvolvido.

---

# 18. Preparação para Produção

## Objetivo

Após a construção, validação e interpretação dos modelos preditivos, foi realizada uma etapa dedicada à preparação para produção, com o objetivo de transformar o protótipo experimental em uma solução capaz de operar de forma segura, reproduzível e escalável em ambientes reais.

Essa etapa contempla aspectos relacionados à persistência dos modelos, validação de dados de entrada, tratamento de situações não previstas durante o treinamento e geração de artefatos necessários para implantação do sistema.

O foco principal consiste em garantir que o modelo possa ser utilizado futuramente em aplicações práticas sem depender do ambiente de desenvolvimento utilizado durante os experimentos.

---

## 18.1 Construção do Pipeline de Produção

Foi desenvolvido um pipeline unificado contendo todos os componentes necessários para execução das previsões.

O pipeline integra:

* modelo preditivo treinado;
* transformações de atributos;
* validações de entrada;
* tratamento de categorias;
* geração de metadados;
* mecanismos de monitoramento.

A estrutura foi encapsulada em uma classe específica denominada **ProductionPipeline**, responsável por centralizar todo o fluxo de inferência.

Essa abordagem reduz a possibilidade de divergências entre o ambiente de treinamento e o ambiente de produção.

---

## 18.2 Persistência e Versionamento

Após a construção do pipeline, todos os componentes foram serializados e armazenados para reutilização futura.

### Artefatos Persistidos

Entre os artefatos salvos destacam-se:

* modelos treinados;
* pipeline completo de produção;
* transformadores de atributos;
* codificadores categóricos;
* listas de variáveis utilizadas;
* resultados experimentais;
* métricas de avaliação.

O pipeline completo foi salvo com sucesso no formato:

```text
production_pipeline.joblib
```

Além disso, o sistema incorpora controle de versão dos artefatos, permitindo rastrear futuras atualizações e facilitar a manutenção da solução.

---

## 18.3 Teste de Inferência

Após a persistência do pipeline, foi realizado um teste de carregamento e execução utilizando exemplos reais do conjunto de teste.

O processo validou:

1. carregamento correto dos artefatos;
2. reconstrução completa do pipeline;
3. geração de previsões;
4. retorno de metadados associados às inferências.

### Exemplo de Inferência

| Valor Real | Valor Previsto |
| ---------: | -------------: |
|         23 |           59.7 |
|         74 |           72.4 |
|         13 |           22.9 |
|         11 |           28.7 |
|         89 |           42.1 |

O objetivo desse teste não foi avaliar a precisão do modelo — já analisada nas etapas anteriores — mas verificar a integridade operacional do pipeline após sua serialização e recarga.

Os resultados confirmaram que o sistema permanece funcional mesmo após o processo completo de persistência.

---

## 18.4 Validação Automática de Entradas

Uma das funcionalidades implementadas foi a validação automática dos dados recebidos pelo sistema.

Essa camada de proteção verifica:

* presença das colunas obrigatórias;
* tipos de dados esperados;
* consistência estrutural;
* categorias desconhecidas;
* possíveis inconsistências de entrada.

Esse mecanismo reduz significativamente o risco de falhas durante a utilização do modelo em ambientes reais.

---

## 18.5 Tratamento de Categorias Desconhecidas

Em aplicações reais é comum que novos valores categóricos surjam após o treinamento do modelo.

Para lidar com esse cenário, foi implementado um mecanismo de detecção automática de categorias não observadas anteriormente.

Durante os testes foram identificados os seguintes exemplos:

```text
club: West Ham United
Season: 24/25
```

Essas categorias não estavam presentes no conjunto de treinamento original.

Mesmo assim, o pipeline permaneceu operacional, emitindo alertas informativos sem interromper a geração das previsões.

Esse comportamento é particularmente importante em aplicações esportivas, onde novos clubes, temporadas e atletas surgem continuamente.

---

## 18.6 Metadados e Monitoramento

Além das previsões, o pipeline retorna informações auxiliares para monitoramento do processo de inferência.

Entre os metadados produzidos estão:

* número de amostras processadas;
* média das previsões;
* mediana das previsões;
* desvio-padrão das previsões;
* alertas gerados durante a execução.

Essas informações podem ser utilizadas futuramente para:

* monitoramento de qualidade;
* detecção de drift;
* auditoria de previsões;
* acompanhamento operacional do sistema.

---

## 18.7 Benefícios para Implantação

A estrutura desenvolvida oferece diversas vantagens para utilização em ambiente produtivo.

Entre os principais benefícios destacam-se:

* reprodutibilidade dos resultados;
* facilidade de implantação;
* redução de erros operacionais;
* escalabilidade da solução;
* simplificação da manutenção;
* maior robustez frente a dados não previstos.

Além disso, a centralização de todo o fluxo em um único pipeline reduz significativamente a complexidade necessária para integração com APIs, dashboards ou sistemas corporativos.

---

## Conclusão da Seção

A etapa de preparação para produção demonstrou que o modelo desenvolvido não se limita a um experimento acadêmico isolado, mas possui estrutura técnica compatível com futuras aplicações práticas.

Os testes realizados confirmaram a capacidade do sistema de carregar artefatos persistidos, executar inferências, validar entradas, identificar categorias desconhecidas e gerar metadados operacionais de forma automática.

Dessa forma, o projeto avança além da simples construção de modelos preditivos, incorporando elementos fundamentais para sua utilização em cenários reais de Sports Analytics e apoio à tomada de decisão em departamentos médicos e técnicos do futebol profissional.

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

O modelo LightGBM Baseline apresentou o melhor desempenho global do projeto, sendo selecionado como modelo final segundo a métrica principal de avaliação (MAE).

---

## Resultados Finais

| Métrica | Valor |
|----------|---------:|
| MAE | 19.95 dias |
| RMSE | 33.28 dias |
| R² | 0.5487 |
| MedianAE | 11.99 dias |

---

## Principais Variáveis Explicativas

A análise SHAP identificou como principais fatores associados ao tempo de recuperação:

1. Injury_target_enc
2. club_target_enc
3. player_injury_rate_percentile
4. Injury_freq
5. days_to_season_end
6. player_age
7. days_since_last_injury
8. rolling_mean_days_3

Essas variáveis demonstram que o histórico médico do atleta, o tipo da lesão e fatores contextuais relacionados ao clube e à temporada exercem influência significativa sobre o tempo de afastamento.

---

## Principais Achados

Os resultados obtidos indicam que:

- o tipo da lesão é o principal fator determinante da duração da recuperação;
- o histórico acumulado de lesões aumenta significativamente o poder preditivo do modelo;
- variáveis temporais contribuem para capturar padrões sazonais importantes;
- a utilização de técnicas avançadas de Feature Engineering gerou ganhos superiores aos obtidos apenas por otimização de hiperparâmetros.

---

## Limitações

Entre as principais limitações observadas destacam-se:

- ausência de informações médicas detalhadas;
- inexistência de métricas fisiológicas dos atletas;
- baixa frequência de algumas lesões graves;
- presença de eventos extremos com longos períodos de recuperação.

---

## Considerações Finais

O projeto demonstrou que é possível prever a duração de lesões esportivas utilizando técnicas modernas de Machine Learning, alcançando desempenho competitivo e elevada interpretabilidade.

Além das previsões pontuais, foram incorporadas análises de explicabilidade, quantificação de incerteza, regressão quantílica e validações temporais, tornando a solução metodologicamente robusta e potencialmente aplicável em cenários reais de Sports Analytics e Medicina Esportiva.

---

# 21. Salvamento de Artefatos e Encerramento do Projeto

## Objetivo

Garantir a persistência completa dos modelos treinados, métricas, configurações e resultados produzidos durante o desenvolvimento do projeto.

---

## Artefatos Salvos

### Modelos

Foram armazenados os seguintes modelos:

- best_model.joblib
- lgbm_optuna.joblib
- catboost_optuna.joblib
- xgboost_optuna.joblib

O arquivo `best_model.joblib` corresponde automaticamente ao modelo com melhor desempenho no benchmark completo.

No experimento final, o modelo salvo foi:

```text
[Baseline] LightGBM
```

---

### Resultados

Foram persistidos:

- results_summary.json
- benchmark_completo.csv
- feature_names.json

---

### Visualizações

Foram armazenadas todas as figuras produzidas ao longo do projeto:

- SHAP
- Benchmark
- Residual Analysis
- Calibration
- Quantile Regression
- Feature Importance

---

### MLflow

Também foram registrados:

- parâmetros;
- métricas;
- artefatos;
- configurações dos experimentos.

---

## Benefícios

O salvamento sistemático dos artefatos garante:

- reprodutibilidade completa;
- auditoria dos experimentos;
- reutilização dos modelos;
- facilidade de implantação em produção;
- rastreabilidade científica.

---

## Encerramento

Ao final da execução, o projeto produz automaticamente:

- melhor modelo global;
- modelos otimizados;
- métricas consolidadas;
- resultados experimentais;
- visualizações;
- logs de execução.

Essa estrutura permite continuidade futura do projeto e facilita sua evolução para ambientes reais de produção.
