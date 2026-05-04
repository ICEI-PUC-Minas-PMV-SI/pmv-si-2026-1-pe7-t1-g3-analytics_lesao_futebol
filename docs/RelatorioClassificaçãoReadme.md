Relatório de Classificação:

O Relatório de Classificação funciona como um diagnóstico minucioso para validar a eficácia de modelos de machine learning. Em vez de entregar um número isolado, ele disseca a performance para cada categoria, permitindo distinguir entre um acerto acidental e uma previsão de qualidade.

Métricas de Desempenho por Classe:
Precisão (Precision): Mede a assertividade das previsões positivas. Responde à pergunta: "Das vezes que o modelo classificou como X, quantas eram realmente X?".

Revocação (Recall): Também chamada de sensibilidade, mede a capacidade de detecção. Indica qual proporção do total de casos reais de uma classe o modelo foi capaz de encontrar.

F1-score: É o balanço ideal, obtido através da média harmônica entre precisão e recall. É a métrica de referência quando há desequilíbrio entre as classes (ex: muito mais jogadores saudáveis do que lesionados).

Suporte (Support): Indica a frequência real de cada classe no banco de dados de teste, servindo de base para o cálculo das outras métricas.

Visão Consolidada (Métricas Agregadas):
Acurácia (Accuracy): Representa o sucesso global do modelo, ou seja, o percentual de predições corretas sobre o total de casos.

Média Macro (Macro Average): Avalia o desempenho médio tratando todas as classes de forma igualitária, independentemente de quantas instâncias cada uma possui.

Média Ponderada (Weighted Average): Calcula o desempenho médio levando em conta o peso proporcional de cada classe, sendo influenciada majoritariamente pelas categorias mais frequentes.

Código em python para gerar o relatório de classificação: [Script Relatório de Classificação](RltClass.py)

Resultado:
<img width="681" height="328" alt="image" src="https://github.com/user-attachments/assets/d6f42e01-5082-4c60-b228-dba0bc9115d3" />

Análise: 

1. Desempenho por Classe:
Classe 0 (Baixo Risco / Lesões Leves): O modelo é praticamente perfeito neste ponto. Com 100% de Precisão e Recall, ele identifica com total segurança quando uma lesão não será grave. Isso é comum porque essa é a classe majoritária (892 casos).

Classe 1 (Alto Risco / Lesões Graves): 

Precisão de 91%: Quando o modelo aponta uma lesão grave, ele acerta em 9 de cada 10 vezes. 

Recall de 86%: O modelo "deixou passar" 14% das lesões graves, classificando-as erroneamente como leves. Em um cenário de alto rendimento, esse é o risco que o departamento médico quer minimizar.

2. O Equilíbrio (F1-Score e Support)
F1-Score (0.88): Para a classe de alto risco, um F1 de 0.88 é muito sólido. Ele mostra que o modelo não está apenas "chutando" que tudo é lesão leve para acertar a acurácia global; ele realmente aprendeu padrões que distinguem a gravidade.

Support (892 vs 77): Há um desbalanceamento de classes. Existem muito mais dados de lesões curtas do que de lesões longas. Isso explica por que as médias "Weighted" (ponderadas) são altas, enquanto a "Macro" (média simples) é um pouco menor (0.93).

