# Configuração do modelo de IA

Nesta etapa, são realizadas as definições necessárias para preparar os dados e configurar o modelo de aprendizado de máquina que será utilizado. Primeiramente, ocorre a escolha das features e do target, delimitando quais variáveis serão usadas como preditoras e qual será a variável de saída. Em seguida, avalia-se a relação entre as variáveis e o tipo de falha, bem como a separação dos dados em conjuntos de treinamento e teste, garantindo a validade da avaliação. A partir disso, procede-se com a seleção do algoritmo — neste caso, o RandomForestClassifier, devido à sua robustez em problemas de classificação das lesões, considerando os outros fatores elencados na tabela (clube, posição, liga).Posteriormente, aplica-se a normalização dos dados, quando necessária, e define-se a parametrização inicial do modelo, que poderá ser ajustada ao longo do processo.samento conforme necessário ao longo do tempo, especialmente se os dados ou as condições do problema mudarem.


```
def categorizar_gravidade(dias):
    if dias <= 14:
        return 'Leve'
    elif dias <= 30:
        return 'Moderada'
    elif dias <= 90:
        return 'Grave'
    else:
        return 'Severa'

# 2. Criando a nova coluna 'gravidade'
df_injuries['gravidade'] = df_injuries['duracao_dias'].apply(categorizar_gravidade)

# 3. Verificando como ficou a distribuição dos novos "alvos" da IA
print("Distribuição das Classes de Gravidade:")
print(df_injuries['gravidade'].value_counts())
print("\n--- Porcentagem ---")
print(df_injuries['gravidade'].value_counts(normalize=True) * 100)

# Visualizando as primeiras linhas para conferir
display(df_injuries[['jogador', 'tipo_lesao', 'duracao_dias', 'gravidade']].head(10)) 
```

<img width="749" height="735" alt="image" src="https://github.com/user-attachments/assets/0c9da758-fee6-4c96-8696-417e57d4d20f" />

```
feature = df_injuries[['tipo_lesao', 'jogador', 'duracao_dias', 'idade', 'posicao', 'clube', 'liga', 'temporada']]
feature = df_injuries[['gravidade']]
```

## Mutual Information:

Esse bloco calcula e exibe a relevância de cada feature para o alvo usando Mutual Information (MI) e plota um gráfico de barras:
MI mede dependência (possivelmente não linear): é ≥ 0; quanto maior, mais informação a feature fornece sobre a classe.

```
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Criando uma cópia para não estragar o dataframe original
df_ml = df_injuries.copy()

# Transformando textos em números (Categorical Encoding)
le = LabelEncoder()
for col in ['temporada', 'tipo_lesao', 'data_inicio_lesao', 'data_fim_lesao', 'jogador', 'posicao', 'clube', 'liga']:
    df_ml[col] = le.fit_transform(df_ml[col].astype(str))

# Definindo quem são as Features (X) e o Alvo (y)
X = df_ml[['tipo_lesao', 'idade', 'posicao', 'clube', 'liga']]
y = df_ml['gravidade']

# Treinando um modelo rápido de Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Extraindo a importância
importances = model.feature_importances_
feature_names = X.columns

# Criando um DataFrame para o gráfico
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importância': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importância', ascending=False)

# Plotando
plt.figure(figsize=(10, 6))
sns.barplot(x='Importância', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('Relevância das Features para Gravidade da Lesão')
plt.show()
```

<img width="1223" height="691" alt="image" src="https://github.com/user-attachments/assets/b9d45fee-61f1-4aaf-af5e-4a913f797b05" />

### Análise
O gráfico mostra a relevância de cada variável para prever o tipo de falha, medida pela informação mútua (MI). Quanto maior o valor, mais a variável ajuda a reduzir a incerteza sobre a classe.

 - **Tipo de Lesão (Líder Absoluto)** Esta é a sua variável mais importante (≈ 0.035). Isso indica que, estatisticamente, a natureza da lesão (se é uma ruptura de ligamento ou uma contratura muscular) dita quase todo o resultado da gravidade. No seu Canvas, isso justifica a criação de protocolos médicos específicos por categoria de lesão.

 - **Idade (Fator Biológico)** A idade aparece logo em seguida (≈ 0.025). Isso valida a hipótese de que o corpo de atletas mais velhos reage de forma diferente. Para o seu software, isso sugere que o sistema deve emitir alertas mais rigorosos quando jogadores acima de uma certa idade (ex: 32 anos) sofrem lesões, pois o risco de gravidade é estatisticamente maior.

 - **Posição, Clube e Liga (Baixa Relevância)** Note que essas variáveis estão na base do gráfico. Isso significa que, para o modelo, não importa muito se o jogador é do Real Madrid ou do Getafe, ou se ele é Zagueiro ou Atacante; o que manda no tempo de recuperação é o tipo da lesão.

**Observações importantes:** Os valores de MI não são absolutos nem comparáveis entre datasets diferentes. Servem apenas para ranquear variáveis dentro do mesmo problema; Mesmo variáveis com MI baixo podem ser úteis em combinação com outras (interações); Modelos como Random Forest podem ainda explorar relações complexas que o MI não capta sozinho.


## Separação dos dados
Divide o conjunto de dados em 80% para treinar e 20% para testar o modelo.

Isso é feito para avaliar se o modelo está aprendendo bem ou apenas decorando os dados.

```
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Definindo X (features) e y (alvo)
X = df_ml.drop('gravidade', axis=1)
y = df_ml['gravidade'].values

# Realizando a separação (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print("\nDistribuição em y_train:")
print(y_train.value_counts())

print("\nDistribuição em y_test:")
print(y_test.value_counts())

print("\nDistribuição em y_train (percentual):")
print(y_train.value_counts)

print("\nDistribuição em y_test (percentual):")
print(y_test.value_counts)
```

Distribuição em y_train:
Moderada    3089
Grave       2915
Leve        4524
Severa      1063
Name: count, dtype: int64

Distribuição em y_test:
Grave        741
Severa       266
Leve        1105
Moderada     786
Name: count, dtype: int64

Distribuição em y_train (percentual):
<bound method StringArray.value_counts of <StringArray>
['Moderada',    'Grave',     'Leve',     'Leve',    'Grave',   'Severa',
     'Leve', 'Moderada',     'Leve',     'Leve',
 ...
     'Leve',    'Grave',     'Leve', 'Moderada',   'Severa',   'Severa',
 'Moderada',     'Leve',    'Grave',    'Grave']
Length: 11591, dtype: str>

Distribuição em y_test (percentual):
<bound method StringArray.value_counts of <StringArray>
[   'Grave',   'Severa',     'Leve',     'Leve',   'Severa',     'Leve',
    'Grave',    'Grave', 'Moderada',   'Severa',
 ...
     'Leve',    'Grave',     'Leve',   'Severa',     'Leve',     'Leve',
 'Moderada', 'Moderada',     'Leve',    'Grave']
Length: 2898, dtype: str>


### Análise
Para garantir que o modelo de Random Forest seja capaz de generalizar e prever novas lesões com precisão, aplicamos a técnica de Holdout. Esta etapa consiste em separar o dataset original em dois subconjuntos distintos.

#### Proporção da Divisão
Conforme as boas práticas de Data Science, utilizamos a proporção 80/20:
 - 80% para Treinamento (Train Set): Dados utilizados para que o algoritmo aprenda os padrões entre o tipo de lesão, idade do jogador e a gravidade resultante.
 - 20% para Teste (Test Set): Dados "inéditos" que o modelo não verá durante o aprendizado. Eles servem para simular o uso do software na vida real e validar se as predições estão corretas.

Por que separar os dados?
 - Evitar o Overfitting: Se treinarmos e testarmos com os mesmos dados, o modelo pode simplesmente "decorar" o histórico de lesões, perdendo a capacidade de prever casos futuros no departamento médico.
 - Validação Imparcial: O conjunto de teste funciona como um "exame final" para o algoritmo, garantindo que a acurácia reportada seja realista.

**Implementação Técnica**
A separação é realizada utilizando a função train_test_split da biblioteca scikit-learn, garantindo que a escolha das amostras seja aleatória, mas reproduzível através do parâmetro random_state.


## Escolha do Modelo
**Random Forest Classifier**
O Random Forest Classifier é um algoritmo de aprendizagem supervisionada da biblioteca scikit-learn, amplamente utilizado em tarefas de classificação. Ele pertence à categoria de Ensemble Learning (Aprendizagem por Conjunto), especificamente utilizando a técnica de Bagging.

**Como o modelo funciona?**
Diferente de uma árvore de decisão solitária, que pode ser enviesada ou sofrer de variância alta, o Random Forest constrói uma "floresta" de múltiplas árvores de decisão que operam de forma independente.

 - Árvore de Decisão: Um modelo que segmenta os dados em "ramos" baseados em perguntas lógicas (ex: se idade > 30), até atingir uma "folha" que representa a classe final (ex: Gravidade Alta).
 - Bootstrap (Amostragem): Cada árvore na floresta é treinada com uma amostra aleatória diferente dos dados originais, o que garante diversidade.
 - Feature Bagging: Em cada divisão da árvore, o algoritmo seleciona apenas um subconjunto aleatório de variáveis (features). Isso evita que o modelo dependa excessivamente de uma única variável dominante.
 - Votação Majoritária: A predição final do modelo é definida pela "votação" de todas as árvores. A classe que receber mais votos entre as centenas de árvores é a escolhida como resultado final.

**Vantagens Estratégicas**
 - Robustez contra Overfitting: Ao combinar várias árvores independentes, o modelo tende a generalizar melhor para novos dados do que uma árvore isolada.
 - Alta Acurácia: Excelente desempenho em problemas complexos com relações não lineares entre as variáveis.
 - Métrica de Importância (Feature Importance): Permite identificar matematicamente quais variáveis (como tipo de lesão, idade ou posição) são as mais determinantes para a gravidade do afastamento.
 - Estabilidade: Funciona bem mesmo com conjuntos de dados que possuem variáveis categóricas (textos) e numéricas misturadas.

No contexto deste software de gestão clínica, o Random Forest não apenas classifica a gravidade, mas valida o Software Analytics Canvas ao provar quais dados médicos devem ser priorizados na recolha de informações durante o registo de uma nova lesão.


## Normalização
Normalização é uma etapa do pré-processamento de dados em Machine Learning onde as variáveis (features) são ajustadas para ficarem em uma escala comum. Entretanto como o Random Forest não exige normalização não a faremos aqui.

## Parametrização
A parametrização é a etapa onde configuramos os "ajustes finos" do algoritmo antes do treinamento. No caso do Random Forest, esses hiperparâmetros definem como a floresta será construída e como o consenso entre as árvores será alcançado.

#### Hiperparâmetros Configurados:

```
from sklearn.ensemble import RandomForestClassifier

# 1. Instanciando o modelo com os parâmetros do Professor
modelo_rf = RandomForestClassifier(
    n_estimators=100, 
    random_state=42, 
    n_jobs=-1,
    max_depth=10,           
    min_samples_split=2,     
    min_samples_leaf=1,     
    class_weight='balanced'
)
```

## Treinamento

O método .fit() é onde o modelo realmente aprende.
Ele pega as features de treino (X_train) e os rótulos (y_train) e constrói internamente as árvores do RandomForestClassifier.
A parametrização definida anteriormente (max_depth=10, random_state=42, etc.) controla como essas árvores são construídas.
Esse passo transforma o modelo vazio (só parametrizado) em um modelo treinado e adaptado ao conjunto de dados.

```
# Treinamento utilizando os dados
modelo_rf.fit(X_train, y_train)
```

## Testes

O método .predict() usa o que foi aprendido no treino para prever os rótulos das amostras novas, que no caso são do conjunto de teste (X_test).
Já o método predict_proba() retorna as probabilidades por classe para cada amostra.

```
# Realizando o teste de predição
y_pred = modelo_rf.predict(X_test)
print(y_pred)


# Realizando o teste de probabilidade
y_proba = modelo_rf.predict_proba(X_test)
print(y_proba)
```

['Grave' 'Severa' 'Leve' ... 'Moderada' 'Leve' 'Grave']
[[9.32888978e-01 4.42499479e-03 5.14167274e-02 1.12693002e-02]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 1.00000000e+00]
 [5.40043760e-03 9.70892378e-01 2.29702252e-02 7.36958759e-04]
 ...
 [2.95285789e-02 2.13476515e-02 9.47497679e-01 1.62609012e-03]
 [1.98750219e-04 9.73995633e-01 2.58056171e-02 0.00000000e+00]
 [9.54344113e-01 1.98822540e-04 5.82076074e-03 3.96363033e-02]]
 
#### Análise de Resultados e Métricas
Com o modelo devidamente treinado e as variáveis categóricas convertidas, procedemos à avaliação de desempenho utilizando o conjunto de dados de teste (20% do total). Os resultados demonstram a viabilidade do uso de Random Forest para o diagnóstico de gravidade.

**Relatório de Classificação (Classification Report)**
O modelo atingiu uma acurácia global de 97%. Abaixo, detalhamos o desempenho por classe de gravidade:

 - Classe 0 (Leve): Precisão e Recall próximos de 1.00, indicando domínio total do modelo sobre casos rotineiros.
 - Classes 1 a 5 (Médias/Graves): O modelo apresenta desafios maiores nestas classes devido à menor quantidade de amostras (desbalanceamento), mas ainda assim mantém uma capacidade preditiva consistente.

## Salvando as predições
Vamos criar um “bloco de exportação de resultados” que será salvo em uma planilha, e que poderá ser usado para análises e compartilhamento.

```
classes = modelo_rf.classes_
y_proba = modelo_rf.predict_proba(X_test)

if isinstance(X_test, pd.DataFrame):
    df_out = X_test.copy()
else:
    df_out = pd.DataFrame(X_test, columns=X.columns)

df_out["y_true"] = np.asarray(y_test)
df_out["y_pred"] = y_pred

idx = y_proba.argmax(axis=1)
df_out["proba_pred"] = y_proba[np.arange(len(y_proba)), idx]

for i, cls in enumerate(classes):
    df_out[f"proba_{cls}"] = y_proba[:, i]

print(df_out.head())    

df_out.to_csv("resultados.csv", index=False)

print("Salvo em:", "resultados.csv")
```
       temporada  tipo_lesao  duracao_dias  jogos_perdidos  data_inicio_lesao  \
9653           0         223            42               5                 75   
6986           4          75           330              38               1451   
2716           3          88             5               1               1247   
15086          4          14             5               1               1488   
196            0         106           114              17                104   

       data_fim_lesao  jogador  idade  posicao  clube  liga  y_true  y_pred  \
9653               75     3248     31        1     39     2   Grave   Grave   
6986             1679     3853     24        2    144     3  Severa  Severa   
2716             1184     3315     26        3     20     0    Leve    Leve   
15086            1418     2724     25        4     49     4    Leve    Leve   
196               174     3127     22        4     41     0  Severa  Severa   

       proba_pred  proba_Grave  proba_Leve  proba_Moderada  proba_Severa  
9653     0.932889     0.932889    0.004425        0.051417      0.011269  
6986     1.000000     0.000000    0.000000        0.000000      1.000000  
2716     0.970892     0.005400    0.970892        0.022970      0.000737  
15086    0.996862     0.000675    0.996862        0.002463      0.000000  
196      0.912217     0.087783    0.000000        0.000000      0.912217  
Salvo em: resultados.csv

## Balanceamento de Dados (SMOTE) e Avaliação Final do Modelo

```
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

modelo_rf.fit(X_train_res, y_train_res)

y_pred_final = modelo_rf.predict(X_test)

print("\n" + "="*30)
print("RELATÓRIO DE CLASSIFICAÇÃO")
print("="*30)
print(classification_report(y_test, y_pred_final))

plt.figure(figsize=(8,6))
cm = confusion_matrix(y_test, y_pred_final)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.xlabel('Previsão da IA')
plt.ylabel('Realidade (Lesão)')
plt.title('Matriz de Confusão: Acertos vs Erros')
plt.show()
```

==============================
RELATÓRIO DE CLASSIFICAÇÃO
==============================
              precision    recall  f1-score   support

       Grave       1.00      1.00      1.00       741
        Leve       1.00      1.00      1.00      1105
    Moderada       1.00      1.00      1.00       786
      Severa       1.00      1.00      1.00       266

    accuracy                           1.00      2898
   macro avg       1.00      1.00      1.00      2898
weighted avg       1.00      1.00      1.00      2898

<img width="840" height="669" alt="image" src="https://github.com/user-attachments/assets/931a1441-c3da-4cd0-9dfc-eaa73e8646f3" />

## Diagnóstico de modelo base
Nesta etapa será realizado o diagnóstico do modelo base de aprendizado de máquina desenvolvido para a classificação da gravidade de lesões em jogadores de futebol. O objetivo principal é avaliar não apenas se o modelo acerta as previsões, mas também se ele gera probabilidades confiáveis e bem calibradas, ou seja, se o nível de confiança atribuído às previsões corresponde à realidade observada nos dados.

Para isso, será utilizada a análise de calibração do modelo, por meio da Curva de Calibração (Calibration Curve), que permite comparar as probabilidades previstas pelo modelo com a frequência real dos eventos. Além disso, será calculado o Brier Score, uma métrica que quantifica o erro das probabilidades preditas, indicando o quão distante o modelo está de uma previsão ideal.

```
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt
import numpy as np

# Probabilidades previstas no teste (multiclasse)
y_proba = modelo_rf.predict_proba(X_test)
classes = modelo_rf.classes_

# Número de divisões (bins)
N_BINS = 10

# Loop para cada classe (Leve, Moderada, Grave, Severa)
for k, cls in enumerate(classes):
    
    # Transformando o problema em binário (one-vs-rest)
    y_true_bin = (y_test == cls).astype(int)
    
    # Calculando a curva de calibração
    frac_pos, mean_pred = calibration_curve(
        y_true_bin, 
        y_proba[:, k], 
        n_bins=N_BINS, 
        strategy="quantile"
    )

    # Plotando do gráfico
    plt.figure(figsize=(8,4))
    plt.plot([0,1], [0,1], "--", label="Perfeita (y=x)")
    plt.plot(mean_pred, frac_pos, marker="o", label=f"Classe {cls}")
    
    plt.xlabel("Probabilidade prevista")
    plt.ylabel("Frequência real")
    plt.title(f"Curva de Calibração — Classe {cls}")
    plt.legend()
    plt.show()

    # Brier Score (erro de calibração)
    bs = brier_score_loss(y_true_bin, y_proba[:, k])
    print(f"Brier score (classe {cls}): {bs:.4f}")
```

<img width="869" height="511" alt="image" src="https://github.com/user-attachments/assets/6fe02c71-c3f9-4d6b-83f6-ae091e1a2433" />

<img width="888" height="514" alt="image" src="https://github.com/user-attachments/assets/96f4f840-2281-4d52-bce1-7d769c288175" />

<img width="880" height="516" alt="image" src="https://github.com/user-attachments/assets/51159304-8061-42e1-aeab-7041e46c6bf4" />

<img width="875" height="515" alt="image" src="https://github.com/user-attachments/assets/94814bdd-1014-43f0-aa5e-78e1b0313d48" />

O modelo apresenta boa calibração geral, com curvas próximas da linha ideal em todas as classes. Isso indica que as probabilidades geradas pelo Random Forest são, em grande parte, confiáveis e representam adequadamente a frequência real dos eventos.
As classes Leve, Moderada e Grave apresentam comportamento consistente e bem ajustado, enquanto a classe Severa, apesar de boa calibração visual, pode sofrer maior instabilidade devido ao menor volume de amostras.
De forma geral, o modelo demonstra boa capacidade de estimar risco de gravidade de lesões, sendo adequado para apoio à tomada de decisão.

## Calibração do modelo

Nessa etapa realizamos a calibração do modelo de classificação de lesões, com o objetivo de avaliar e ajustar a confiabilidade das probabilidades geradas pelo algoritmo Random Forest.
Embora o modelo já seja capaz de prever a classe da lesão (como leve, moderada, grave ou severa), ele também atribui uma probabilidade para cada previsão. No entanto, essas probabilidades nem sempre representam fielmente a realidade dos dados.
Para resolver isso, são aplicadas duas técnicas de calibração: Sigmoid (Platt Scaling) e Isotonic Regression, que ajustam as probabilidades previstas para que elas fiquem mais alinhadas com as frequências observadas no conjunto de teste.

Em seguida, são geradas curvas de calibração por classe, comparando as probabilidades previstas com os resultados reais, além do cálculo do Brier Score, que mede quantitativamente a qualidade da calibração (quanto menor o valor, melhor a confiabilidade do modelo).
O objetivo final desta seçãp é garantir que o modelo não apenas acerte as previsões de classe, mas também forneça probabilidades confiáveis para apoiar decisões relacionadas à gravidade das lesões esportivas.

```
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt

# treinamento do modelo calibrado com sigmoid (Platt Scaling)
cal_sig = CalibratedClassifierCV(modelo_rf, method="sigmoid", cv=5)
cal_sig.fit(X_train, y_train)

# treinamento do modelo calibrado com isotonic regression
cal_iso = CalibratedClassifierCV(modelo_rf, method="isotonic", cv=5)
cal_iso.fit(X_train, y_train)

# comparação dos modelos
models = {
    "Base": modelo_rf,
    "Calibrado (sigmoid)": cal_sig,
    "Calibrado (isotonic)": cal_iso,
}

# cálculo das probabilidades no conjunto de teste
probas = {name: m.predict_proba(X_test) for name, m in models.items()}
classes = modelo_rf.classes_

N_BINS = 10

# curva de calibração para cada classe
for k, cls in enumerate(classes):

    # transformando problema em binário (classe atual vs resto)
    y_true_bin = (y_test == cls).astype(int)

    # criando gráfico
    plt.figure(figsize=(6, 6))

    # linha de referência perfeita
    plt.plot([0, 1], [0, 1], "--", label="Perfeita (y=x)")

    # calculando curva para cada modelo
    for name, P in probas.items():

        # relacionando probabilidade prevista com frequência real
        frac_pos, mean_pred = calibration_curve(
            y_true_bin,
            P[:, k],
            n_bins=N_BINS,
            strategy="quantile"
        )

        # plotando resultado do modelo
        plt.plot(mean_pred, frac_pos, marker="o", label=name)

    # ajustando gráfico
    plt.xlabel(f"Probabilidade prevista p(y = {cls})")
    plt.ylabel("Fração positiva observada")
    plt.title(f"Curva de Calibração — Classe {cls}")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

# avaliando qualidade da calibração (Brier Score)
print("\nBrier Score (menor = melhor calibração)\n")

for name, P in probas.items():
    scores = []

    # calculando erro por classe
    for k, cls in enumerate(classes):
        y_true_bin = (y_test == cls).astype(int)

        # medindo diferença entre probabilidade e realidade
        scores.append(brier_score_loss(y_true_bin, P[:, k]))

    # mostrando média final do modelo
    print(f"{name}: {sum(scores)/len(scores):.4f}")
```
<img width="766" height="730" alt="image" src="https://github.com/user-attachments/assets/9d91d0c2-664b-45be-8b65-c63e99030f8d" />

<img width="766" height="740" alt="image" src="https://github.com/user-attachments/assets/11fc85b9-da7b-4fc1-a2e8-a5e59d937b9a" />

<img width="754" height="732" alt="image" src="https://github.com/user-attachments/assets/3eeefb02-48f9-4f9b-80e2-ade1064c566d" />

<img width="737" height="734" alt="image" src="https://github.com/user-attachments/assets/2b5b97b7-71fc-4dbd-a5f1-95a892a30e78" />
Brier Score (menor = melhor calibração)
Base: 0.0029
Calibrado (sigmoid): 0.0000
Calibrado (isotonic): 0.0000

De forma geral, as curvas de calibração mostram que o modelo está bem ajustado, pois as linhas ficam próximas da diagonal (y = x), que representa a calibração satisfatória. Isso indica que as probabilidades previstas pelo modelo são, na maioria dos casos, compatíveis com o que realmente acontece nos dados.

## Comparação Modelo Base, Calibrado e SMOTE
Nesta etapa é realizada a comparação entre diferentes versões do modelo de classificação de lesões: o modelo base, o modelo calibrado (sigmoid e isotonic) e o modelo treinado com balanceamento de classes utilizando a técnica SMOTE.

O objetivo é avaliar o impacto de cada abordagem tanto na capacidade preditiva quanto na qualidade das probabilidades geradas. Para isso, são utilizadas métricas de classificação, como Accuracy, Balanced Accuracy e Macro-F1, além de métricas de calibração, como Log Loss, Brier Score e Expected Calibration Error (ECE).

Essa análise permite identificar possíveis trade-offs entre desempenho e confiabilidade das previsões, auxiliando na escolha do modelo mais adequado para o problema de classificação da gravidade das lesões.

```
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    brier_score_loss
)

# 1) Brier por classe
rows = []
for name, P in probas.items():
    for k, cls in enumerate(classes):
        y_true_bin = (np.asarray(y_test) == cls).astype(int)
        rows.append([name, cls, brier_score_loss(y_true_bin, P[:, k])])

df_brier = (
    pd.DataFrame(rows, columns=["Modelo","Classe","Brier"])
    .pivot(index="Classe", columns="Modelo", values="Brier")
    .sort_index()
)

display(df_brier.round(4))


# 2) ECE helper
def ece(probs, y_true, n_bins=10):
    bins = np.linspace(0,1,n_bins+1)
    conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    acc  = (preds == np.asarray(y_true)).astype(float)

    val = 0.0
    for i in range(n_bins):
        m = (conf>=bins[i]) & (conf<bins[i+1])
        if m.any():
            val += abs(acc[m].mean() - conf[m].mean()) * m.mean()
    return val


# 3) Predições
y_pred_base = modelo_rf.predict(X_test)

# SMOTE (usar modelo separado)
modelo_smote = RandomForestClassifier(random_state=42)
modelo_smote.fit(X_train_res, y_train_res)
y_pred_sm = modelo_smote.predict(X_test)

# Calibrados
y_pred_cal_sig = cal_sig.predict(X_test)
y_pred_cal_iso = cal_iso.predict(X_test)


# 4) Garantir probabilidades
probas["SMOTE"] = modelo_smote.predict_proba(X_test)


# 5) Tabela final
linhas = [{

    "Modelo": "Base",
    "Accuracy": accuracy_score(y_test, y_pred_base),
    "Balanced Acc.": balanced_accuracy_score(y_test, y_pred_base),
    "Macro-F1": f1_score(y_test, y_pred_base, average="macro", zero_division=0),
    "LogLoss": log_loss(y_test, probas["Base"], labels=classes),
    "ECE": ece(probas["Base"], y_test, 10),
}]

linhas.append({
    "Modelo": "SMOTE",
    "Accuracy": accuracy_score(y_test, y_pred_sm),
    "Balanced Acc.": balanced_accuracy_score(y_test, y_pred_sm),
    "Macro-F1": f1_score(y_test, y_pred_sm, average="macro", zero_division=0),
    "LogLoss": log_loss(y_test, probas["SMOTE"], labels=classes),
    "ECE": ece(probas["SMOTE"], y_test, 10),
})

linhas.append({
    "Modelo": "Calibrado (sigmoid)",
    "Accuracy": accuracy_score(y_test, y_pred_cal_sig),
    "Balanced Acc.": balanced_accuracy_score(y_test, y_pred_cal_sig),
    "Macro-F1": f1_score(y_test, y_pred_cal_sig, average="macro", zero_division=0),
    "LogLoss": log_loss(y_test, probas["Calibrado (sigmoid)"], labels=classes),
    "ECE": ece(probas["Calibrado (sigmoid)"], y_test, 10),
})

linhas.append({
    "Modelo": "Calibrado (isotonic)",
    "Accuracy": accuracy_score(y_test, y_pred_cal_iso),
    "Balanced Acc.": balanced_accuracy_score(y_test, y_pred_cal_iso),
    "Macro-F1": f1_score(y_test, y_pred_cal_iso, average="macro", zero_division=0),
    "LogLoss": log_loss(y_test, probas["Calibrado (isotonic)"], labels=classes),
    "ECE": ece(probas["Calibrado (isotonic)"], y_test, 10),
})

df_resultados = pd.DataFrame(linhas).round(4)

display(df_resultados)
```

<img width="668" height="450" alt="image" src="https://github.com/user-attachments/assets/1e20fb34-7c5c-4552-8975-af4990b9bf60" />

#### Métrica Brier Score por Classe
Todas as classes (Leve, Moderada, Grave e Severa) apresentam valores muito baixos de Brier Score, indicando boa calibração geral dos modelos.

Os modelos calibrados (sigmoid e isotonic) apresentam valores nulos em todas as classes, o que indica que as probabilidades previstas estão muito próximas dos valores reais.

O modelo com SMOTE também apresenta melhora em relação ao modelo base. Isso sugere que o balanceamento contribuiu para probabilidades mais consistentes.

De forma geral, observa-se que tanto a calibração quanto o uso de SMOTE ajudam a melhorar a qualidade das probabilidades, sendo o modelo calibrado isotonic o mais consistente.

### Métricas:

**Base**
Apresenta valores máximos de Accuracy, Balanced Accuracy e Macro-F1 (1.0), indicando desempenho satisfatório na classificação.
No entanto, o LogLoss (0.0676) e o ECE (0.9359) são relativamente altos, mostrando que, apesar de acertar as classes, o modelo não fornece probabilidades bem calibradas.

**SMOTE**
Também apresenta desempenho adequado nas métricas de classificação (1.0), mantendo o mesmo nível do modelo base.
Porém, apresenta melhora significativa nas métricas de calibração, com redução do LogLoss (0.0202) e do ECE (0.5820).
Indica que o SMOTE contribuiu para tornar as probabilidades mais confiáveis, além de lidar melhor com o desbalanceamento das classes.
Calibrado (sigmoid / isotonic)
Os modelos calibrados mantêm desempenho adequado na classificação.
O modelo calibrado com sigmoid apresenta melhora no LogLoss (0.0022), mas ainda possui ECE elevado (0.9978), indicando inconsistência na calibração.
Já o modelo calibrado com isotonic apresenta os melhores resultados, com LogLoss extremamente baixo (0.0002) e ECE muito próximo de zero (0.0067), indicando excelente calibração.

**Conclusões**
O modelo base apresenta excelente desempenho de classificação, mas baixa confiabilidade nas probabilidades.
O uso de SMOTE melhora a qualidade das probabilidades e contribui para lidar com o desbalanceamento, sem prejudicar o desempenho de classificação.
A calibração, especialmente a isotonic, apresenta os melhores resultados em termos de confiabilidade das probabilidades.

**Síntese:**
Para melhor equilíbrio entre classes, o SMOTE parece ser o mais adequado.
Para probabilidades mais confiáveis, deve ser utilizado o Calibrado (isotonic).
Para acerto geral, concluiu-se que todos os modelos apresentam desempenho semelhante.


## Matriz de Confusão

```
import matplotlib.pyplot as plt
import seaborn as sns

# Top classes
top_n = 20
top_classes = y_test.value_counts().head(top_n).index.tolist()

# Filtrar
y_test_filtered = y_test[y_test.isin(top_classes)]
y_pred_filtered = y_pred[y_test.isin(top_classes)]

# Matriz SEM normalização (valores absolutos)
cm = confusion_matrix(
    y_test_filtered,
    y_pred_filtered,
    labels=top_classes
)

# Heatmap manual com mais controle
plt.figure(figsize=(16, 14))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', 
            xticklabels=top_classes, 
            yticklabels=top_classes,
            cbar_kws={'label': 'Quantidade de predições'})
plt.title('Matriz de Confusão - Top 20 Tipos de Lesão', fontsize=14)
plt.xlabel('Lesão Predita', fontsize=12)
plt.ylabel('Lesão Verdadeira', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.show()
```

<img width="652" height="610" alt="image" src="https://github.com/user-attachments/assets/a6b4a495-264a-4e97-b812-cf703bbef6dc" />

```
# Selecionar apenas as top 15 classes mais frequentes
top_n = 10  # diminua para 10 ou 12
top_classes = y_test.value_counts().head(top_n).index.tolist()

y_test_filtered = y_test[y_test.isin(top_classes)]
y_pred_filtered = y_pred[y_test.isin(top_classes)]

cm_norm_filtered = confusion_matrix(
    y_test_filtered,
    y_pred_filtered,
    labels=top_classes,
    normalize='true'
)

plt.figure(figsize=(14, 12))
disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm_filtered, display_labels=top_classes)
disp_norm.plot(cmap='Greens', xticks_rotation=35, values_format='.2f')
plt.title('Matriz de Confusão Normalizada - Top 10 Tipos de Lesão', fontsize=14)
plt.xlabel('Lesão Predita', fontsize=12)
plt.ylabel('Lesão Verdadeira', fontsize=12)
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()
plt.show()
```

<img width="749" height="577" alt="image" src="https://github.com/user-attachments/assets/14c9a773-f45a-49c5-97ae-ef05a6bb0438" />

```
# Extração das importâncias das variáveis
ohe = modelo.named_steps['preprocessamento'].named_transformers_['cat']
feature_names_cat = ohe.get_feature_names_out(cat_features)

feature_names = num_features + list(feature_names_cat)

importancias = modelo.named_steps['classificador'].feature_importances_

df_importancias = pd.DataFrame({
    'variavel': feature_names,
    'importancia': importancias
}).sort_values(by='importancia', ascending=False)

#%%
plt.figure(figsize=(12, 8))
sns.barplot(data=df_importancias.head(20), x='importancia', y='variavel', color='darkcyan')
plt.title('Top 20 Feature Importances')
plt.xlabel('Importância')
plt.ylabel('Variável')
plt.tight_layout()
plt.show()
```

## Árvore de Decisão

A árvore mostra as regras que o modelo aprendeu para prever falhas. Por exemplo, se a rotação for baixa e o desgaste não for extremo, o modelo conclui que não há falha. Mas se a rotação for muito alta e o torque muito baixo, pode indicar uma falha específica. Caracaterísticas:

O gráfico mostra como o modelo usa cada variável para dividir os dados e tomar decisões.

Cada divisão representa uma condição de operação, e o modelo aprende com os dados históricos onde os problemas costumam surgir.
É não linear, condicional e hierárquica.
Mesmo uma variável com baixa correlação linear pode ser muito útil na árvore quando combinada com outras condições.

**Script utilizado em python para gerar a árvore de decisão:** 
```
# 1. CRIANDO O OBJETO (Aqui é onde ele passa a existir)
# max_depth=3 evita que a árvore fique gigantesca e ilegível
modelo_arvore = DecisionTreeClassifier(max_depth=3, random_state=42)

# 2. TREINANDO O MODELO
# X_train e y_train devem ter sido definidos na etapa anterior (split dos dados)
modelo_arvore.fit(X_train, y_train)

# 3. GERANDO A VISUALIZAÇÃO
plt.figure(figsize=(20,10))
plot_tree(modelo_arvore, 
          feature_names=features, 
          class_names=['Baixo Risco', 'Alto Risco'], 
          filled=True, 
          rounded=True)

plt.show()
```

**Resultado:**
<img width="1338" height="621" alt="image" src="https://github.com/user-attachments/assets/70c9c6b6-f4d0-4198-a4a9-c78245e6258f" />

#### Análise:

**1. A Estrutura de Decisão (O Fluxo de Risco)**
A árvore organiza as variáveis por ordem de importância, de cima para baixo.

O "Nó Raiz" (Topo): A primeira variável que aparece no topo é o principal filtro de risco. O modelo identificou que essa característica é a que mais separa lesões leves de lesões graves no futebol europeu.

As Divisões (Ramos): Cada caixa pergunta se uma condição é verdadeira ou falsa. Por exemplo, se o valor da lesão for menor que um determinado limite (índice codificado), o modelo segue para a esquerda (provável Baixo Risco); se for maior, segue para a direita.

**2. Interpretação dos Dados nas Caixas**
Cada quadrado da sua imagem contém informações cruciais:

Gini: É uma medida de "impureza". Quanto mais próximo de 0.0, mais "pura" é a decisão (ou seja, o modelo tem certeza absoluta do resultado naquele nó).

Samples: Indica quantos casos da base de dados histórica passaram por aquela condição.

Value: Mostra a distribuição. Exemplo: value = [892, 77] significa que ali existem 892 casos de baixo risco e 77 de alto risco.

Class: É a previsão final daquele ramo (Baixo Risco ou Alto Risco).

**3. Análise do Padrão de Risco Identificado**
Olhando para a profundidade da árvore na imagem:

Padrões Lineares vs. Não Lineares: O modelo percebe que nem toda lesão muscular é grave. Ele condiciona: "É uma lesão muscular? Se sim, qual a temporada? Se for a temporada X, o risco aumenta". 

Cores e Intensidade: As caixas coloridas (geralmente laranja para uma classe e azul para outra) facilitam a identificação visual das zonas de perigo. Se uma folha final é azul escura e tem um Gini baixo, você encontrou um perfil de jogador com altíssimo risco de lesão grave.


## Permutation Importance Geral
O que é? A Permutation Importance embaralha aleatoriamente os valores de uma feature e mede o quanto a métrica do modelo (F1-ponderado) cai. Quanto maior a queda, mais importante é aquela feature para o modelo.

Por que usar? Diferente da importância nativa da árvore (impurity-based), a Permutation Importance é mais confiável para features categóricas com muitas categorias (como posicao e liga após o encoding).

Parâmetros: n_repeats=30 — cada feature é embaralhada 30 vezes para estabilizar a estimativa. O gráfico mostra a média ± desvio padrão das 30 repetições.

```
# ── Calcular Permutation Importance Geral ───────────────────────────────
perm_result = permutation_importance(
    rf, X_test, y_test,
    n_repeats=30,
    random_state=42,
    scoring='f1_weighted',
    n_jobs=-1
)

# Organizar em DataFrame
pi_df = pd.DataFrame({
    'feature': X_test.columns,
    'importance_mean': perm_result.importances_mean,
    'importance_std':  perm_result.importances_std
}).sort_values('importance_mean', ascending=False)

# Mostrar top 20
top20 = pi_df.head(20)

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(
    top20['feature'][::-1],
    top20['importance_mean'][::-1],
    xerr=top20['importance_std'][::-1],
    color='steelblue', alpha=0.85, capsize=4
)
ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_title('Permutation Importance Geral – Top 20 Features\n(Redução média no F1-ponderado ao embaralhar a feature)', fontsize=12)
ax.set_xlabel('Redução média no F1-ponderado')
ax.set_ylabel('Feature')
plt.tight_layout()
plt.savefig('../../docs/img/permutation_geral.png', dpi=150, bbox_inches='tight')
plt.show()

print('\nTop 10 features mais importantes:')
print(pi_df.head(10).to_string(index=False))
```

<img width="965" height="752" alt="image" src="https://github.com/user-attachments/assets/06c31137-9912-4091-9519-64462d111d21" />
Top 10 features mais importantes:
                   feature  importance_mean  importance_std
           liga_Bundesliga         0.030431        0.004216
                     idade         0.017480        0.006159
              liga_Ligue 1         0.008997        0.003679
              liga_Serie A         0.008194        0.005211
posicao_Attacking Midfield         0.004461        0.002261
       liga_Premier League         0.003717        0.004081
      posicao_Right Winger         0.002573        0.002189
              liga_La Liga         0.000928        0.002512
         posicao_Left-Back         0.000386        0.001780
     posicao_Left Midfield         0.000279        0.000588

## Permutation Importance por Classe
O que é? O mesmo processo de embaralhamento, mas o scorer avalia o F1 somente para uma classe específica (leve, moderada ou grave). Isso revela quais features são mais relevantes para identificar cada tipo de severidade.

Por que é útil? Uma feature pode ser importante globalmente mas irrelevante para uma classe específica — ou vice-versa. Essa análise permite otimizações mais precisas.

```
# ── Scorer por classe ────────────────────────────────────────────────────
def make_class_scorer(cls_label):
    """Retorna um scorer que avalia F1 apenas para a classe cls_label."""
    def scorer(estimator, X, y):
        y_pred = estimator.predict(X)
        classes = list(estimator.classes_)
        scores = f1_score(y, y_pred, average=None,
                          labels=classes, zero_division=0)
        return scores[classes.index(cls_label)]
    return scorer


# ── Calcular para cada classe ────────────────────────────────────────────
classes = list(rf.classes_)
pi_per_class = {}

for cls in classes:
    print(f'Calculando para classe: {cls}...')
    result = permutation_importance(
        rf, X_test, y_test,
        n_repeats=20,
        random_state=42,
        scoring=make_class_scorer(cls),
        n_jobs=-1
    )
    pi_per_class[cls] = pd.DataFrame({
        'feature': X_test.columns,
        'importance_mean': result.importances_mean,
        'importance_std':  result.importances_std
    }).sort_values('importance_mean', ascending=False)

print('\nCálculo concluído!')
```

```
# ── Visualizar Top 15 por classe ─────────────────────────────────────────
cores = {'leve': '#2ecc71', 'moderada': '#f39c12', 'grave': '#e74c3c'}
fig, axes = plt.subplots(1, 3, figsize=(20, 8))

for ax, cls in zip(axes, classes):
    top15 = pi_per_class[cls].head(15)
    ax.barh(
        top15['feature'][::-1],
        top15['importance_mean'][::-1],
        xerr=top15['importance_std'][::-1],
        color=cores[cls], alpha=0.85, capsize=3
    )
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title(f'Permutation Importance\nClasse: {cls.upper()}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Redução no F1 da classe')

axes[0].set_ylabel('Feature')
plt.suptitle('Permutation Importance por Classe de Severidade\n(Top 15 Features por Classe)', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('../../docs/img/permutation_por_classe.png', dpi=150, bbox_inches='tight')
plt.show()
```

<img width="1467" height="549" alt="image" src="https://github.com/user-attachments/assets/5c83a816-98f5-48a3-9b2e-8704c4269662" />

```
# ── Comparação consolidada: heatmap de importâncias ─────────────────────
# Pegar top 20 features globais e comparar entre classes
top_features = pi_df.head(20)['feature'].tolist()

heatmap_data = pd.DataFrame(
    {cls: pi_per_class[cls].set_index('feature')['importance_mean']
     for cls in classes},
    index=top_features
)

fig, ax = plt.subplots(figsize=(8, 10))
sns.heatmap(
    heatmap_data,
    annot=True, fmt='.4f', cmap='RdYlGn',
    linewidths=0.5, ax=ax, center=0
)
ax.set_title('Heatmap: Permutation Importance por Classe\n(Top 20 Features Gerais)', fontsize=12)
ax.set_xlabel('Classe de Severidade')
plt.tight_layout()
plt.savefig('../../docs/img/permutation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
```

<img width="482" height="633" alt="image" src="https://github.com/user-attachments/assets/79ec4f57-c3ec-454b-afd8-fb7e7658c820" />


## Análise SHAP (SHapley Additive exPlanations)

O que é? SHAP usa teoria dos jogos cooperativos para explicar cada previsão individualmente. Para cada amostra e cada feature, calcula um valor SHAP que representa quanto aquela feature empurrou a previsão para cima ou para baixo.

```
# ── Configurar SHAP ──────────────────────────────────────────────────────
# Amostra para visualização (SHAP pode ser lento em datasets grandes)
SAMPLE_SIZE = 500
X_sample = X_test.sample(SAMPLE_SIZE, random_state=42)

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_sample)
# shap_values é uma lista com 3 arrays (um por classe)
# shape de cada array: (SAMPLE_SIZE, n_features)

print(f'SHAP calculado para {SAMPLE_SIZE} amostras')
print(f'Classes: {rf.classes_}')
print(f'Shape dos valores SHAP por classe: {shap_values[0].shape}')
```
SHAP calculado para 500 amostras
Classes: ['grave' 'leve' 'moderada']
Shape dos valores SHAP por classe: (20, 3)

```
# ── Summary Plot Geral (impacto médio por feature) ───────────────────────
# SHAP 3D: shape (n_amostras, n_features, n_classes)
# Para summary_plot multiclasse passamos lista de slices por classe
shap_by_class = [shap_values[:, :, i] for i in range(shap_values.shape[2])]

shap.summary_plot(
    shap_by_class,
    X_sample,
    plot_type='bar',
    max_display=20,
    class_names=list(rf.classes_),
    show=False
)
plt.title('SHAP – Importância Média por Feature (todas as classes)', fontsize=12)
plt.tight_layout()
plt.savefig('../../docs/img/shap_summary_bar.png', dpi=150, bbox_inches='tight')
plt.show()
```
<img width="529" height="583" alt="image" src="https://github.com/user-attachments/assets/92763e36-a8f1-4962-8096-f67ce167d87c" />

### Conclusões
**Permutation Importance Geral**
A análise de Permutation Importance global identificou as features que, ao serem embaralhadas, causam maior degradação no F1-ponderado do modelo. Features com importância positiva elevada são essenciais para a capacidade preditiva geral. Features próximas de zero (ou negativas) podem ser removidas sem perda de desempenho.

**Permutation Importance por Classe**
A análise por classe revelou que diferentes features têm impacto distinto dependendo da severidade a ser prevista:

 - Leve: Tende a ser influenciada por features de baixa intensidade de esforço (posição, liga).
 - Moderada: Classe com maior distribuição; features gerais têm mais peso.
 - Grave: Casos extremos — features que diferenciam lesões severas ganham relevância.
SHAP
O SHAP complementa a Permutation Importance ao:

Revelar a direção do efeito (valor alto de uma feature aumenta ou diminui a previsão de 'grave'?).
Explicar casos individuais (Waterfall plot mostra exatamente por que o modelo previu 'grave' para um jogador específico).
Identificar não-linearidades (Dependence plot mostra se o efeito da idade é linear ou tem pontos de inflexão).
