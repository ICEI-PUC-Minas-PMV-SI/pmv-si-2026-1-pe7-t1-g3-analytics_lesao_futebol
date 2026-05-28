# Injury Prediction Advanced

**Predição de Duração de Lesões no Futebol Europeu — Tese de Mestrado**

Framework modular profissional para predição de duração de lesões utilizando técnicas avançadas de Machine Learning.

## Estrutura do Projeto

```
injury_prediction_advanced/
├── src/
│   ├── __init__.py              # Package init
│   ├── config.py                # Configurações globais, seeds, constantes
│   ├── data_processing.py       # Carregamento, limpeza, split temporal
│   ├── feature_engineering.py   # Features históricas, temporais, estatísticas
│   ├── models.py                # Ensemble avançado + Optuna optimisation
│   ├── evaluation.py            # Métricas, nested CV, análise estatística
│   ├── explainability.py        # SHAP interpretabilidade
│   ├── calibration.py           # Intervalos de predição, conformal prediction
│   ├── quantile_models.py       # Regressão quantílica (cenários)
│   ├── visualization.py         # Gráficos profissionais para tese
│   ├── experiment_tracking.py   # MLflow tracking
│   ├── production.py            # Pipeline de produção
│   └── utils.py                 # Funções auxiliares
├── notebooks/                   # Jupyter notebooks
├── models/                      # Modelos serializados
├── figures/                     # Visualizações geradas
├── mlruns/                      # MLflow artifacts
├── logs/                        # Logs de execução
└── README.md
```

## Melhorias Implementadas

| # | Melhoria | Módulo |
|---|----------|--------|
| 1 | Causalidade Temporal | `feature_engineering.py` (TemporalValidator) |
| 2 | Features Históricas Avançadas | `feature_engineering.py` (PlayerHistoryFeatures) |
| 3 | Features Temporais/Calendário | `feature_engineering.py` (TemporalFeatures) |
| 4 | Feature Engineering Estatístico | `feature_engineering.py` (StatisticalFeatures) |
| 5 | Ensemble Avançado | `models.py` (AdvancedEnsemble) |
| 6 | Otimização Optuna | `models.py` (OptunaOptimizer) |
| 7 | SHAP Explainability | `explainability.py` (SHAPExplainer) |
| 8 | Calibração & Incerteza | `calibration.py` (UncertaintyEstimator) |
| 9 | Error Analysis | `evaluation.py` (ModelEvaluator) |
| 10 | Regressão Quantílica | `quantile_models.py` (QuantileRegression) |
| 11 | Nested Cross-Validation | `evaluation.py` (nested_cross_validation) |
| 13 | Análise Estatística | `evaluation.py` (bootstrap_confidence_intervals, paired_t_test) |
| 15 | Experiment Tracking | `experiment_tracking.py` (MLflowTracker) |
| 17 | Visualizações Profissionais | `visualization.py` |
| 18 | Cenários Reais | `evaluation.py` (ScenarioEvaluator) |
| 19 | Pipeline de Produção | `production.py` (ProductionPipeline) |

## Uso Rápido

```python
import sys
sys.path.insert(0, '/home/ubuntu/injury_prediction_advanced')

from src.config import *
from src.utils import set_global_seed, create_directories
from src.data_processing import load_data, clean_data, temporal_split
from src.feature_engineering import FeaturePipeline
from src.models import get_base_models, AdvancedEnsemble, OptunaOptimizer
from src.evaluation import (
    regression_metrics, ModelEvaluator, nested_cross_validation,
    bootstrap_confidence_intervals, ScenarioEvaluator
)
from src.explainability import SHAPExplainer
from src.calibration import UncertaintyEstimator
from src.quantile_models import QuantileRegression
from src.visualization import *
from src.production import ProductionPipeline

# Inicializar
set_global_seed()
create_directories()

# Pipeline
df = load_data()
df = clean_data(df)
train_df, test_df = temporal_split(df)

pipeline = FeaturePipeline()
train_df = pipeline.fit_transform(train_df)
test_df = pipeline.transform(test_df)

features = pipeline.get_feature_names()
```

## Validações de Causalidade Temporal

Todas as features históricas são calculadas usando **exclusivamente dados passados**:
- `PlayerHistoryFeatures`: usa `shift(1)` e loops com `[:i]`
- `SafeFrequencyEncoder`: fitted apenas no treino
- `SafeTargetEncoder`: fitted apenas no treino com smoothing
- `StatisticalFeatures`: normalização fitted no treino
- `GroupKFold` por `player_name` em todas as validações cruzadas

## Dependências

```
numpy pandas scikit-learn matplotlib seaborn
catboost xgboost lightgbm
shap optuna mlflow
scipy joblib
```

## Dataset

- **15.603 registros** de lesões de 5 ligas europeias (2020-2025)
- **11 colunas** originais + features engenheiradas
- **Target:** `days_num` (duração da lesão em dias)
