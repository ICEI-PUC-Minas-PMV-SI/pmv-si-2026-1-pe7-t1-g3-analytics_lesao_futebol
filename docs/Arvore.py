from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# 1. Carregar os dados (A variável 'df' nasce aqui)
file_path = "full_dataset_thesis - 1.csv"
df = kagglehub.dataset_load(
  KaggleDatasetAdapter.PANDAS,
  "sananmuzaffarov/european-football-injuries-2020-2025",
  file_path,
)

# 2. Tratamento de dados (Agora o 'df' já existe)
# Extraímos os números da coluna 'Days' e criamos o alvo de classificação
df['Days_num'] = df['Days'].str.extract('(\d+)').astype(float)
df['target_gravidade'] = (df['Days_num'] > 28).astype(int)

# 3. Preparação das variáveis para o modelo
# Usando colunas que existem no seu dataset
features = ['Injury', 'Season']
X = df[features].copy()
y = df['target_gravidade']

# Transformar textos em números para o modelo entender
le = LabelEncoder()
for col in X.columns:
    X[col] = le.fit_transform(X[col].astype(str))

# 4. Divisão entre Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Treinar o modelo
modelo = RandomForestClassifier(random_state=42)
modelo.fit(X_train, y_train)

# 6. Gerar o Relatório de Classificação no Terminal
y_pred = modelo.predict(X_test)
print("\n" + "="*40)
print("RELATÓRIO DE CLASSIFICAÇÃO")
print("="*40)
print(classification_report(y_test, y_pred, target_names=['Leve/Média', 'Grave (>28 dias)']))


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
