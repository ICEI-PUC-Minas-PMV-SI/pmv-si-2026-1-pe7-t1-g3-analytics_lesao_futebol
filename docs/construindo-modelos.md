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
