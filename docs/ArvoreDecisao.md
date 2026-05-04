A árvore mostra as regras que o modelo aprendeu para prever falhas. Por exemplo, se a rotação for baixa e o desgaste não for extremo, o modelo conclui que não há falha. Mas se a rotação for muito alta e o torque muito baixo, pode indicar uma falha específica. Caracaterísticas:

O gráfico mostra como o modelo usa cada variável para dividir os dados e tomar decisões.

Cada divisão representa uma condição de operação, e o modelo aprende com os dados históricos onde os problemas costumam surgir.

É não linear, condicional e hierárquica.

Mesmo uma variável com baixa correlação linear pode ser muito útil na árvore quando combinada com outras condições.

Script utilizado em python para gerar a árvore de decisão: [Script Árvore de decisão](Arvore.py)

Resultado:
<img width="1919" height="1018" alt="image" src="https://github.com/user-attachments/assets/e4ebcac7-1c20-47f8-ae03-eccaa72a84ea" />

Análise:

1. A Estrutura de Decisão (O Fluxo de Risco)
A árvore organiza as variáveis por ordem de importância, de cima para baixo.

O "Nó Raiz" (Topo): A primeira variável que aparece no topo é o principal filtro de risco. O modelo identificou que essa característica é a que mais separa lesões leves de lesões graves no futebol europeu.

As Divisões (Ramos): Cada caixa pergunta se uma condição é verdadeira ou falsa. Por exemplo, se o valor da lesão for menor que um determinado limite (índice codificado), o modelo segue para a esquerda (provável Baixo Risco); se for maior, segue para a direita.

2. Interpretação dos Dados nas Caixas
Cada quadrado da sua imagem contém informações cruciais:

Gini: É uma medida de "impureza". Quanto mais próximo de 0.0, mais "pura" é a decisão (ou seja, o modelo tem certeza absoluta do resultado naquele nó).

Samples: Indica quantos casos da base de dados histórica passaram por aquela condição.

Value: Mostra a distribuição. Exemplo: value = [892, 77] significa que ali existem 892 casos de baixo risco e 77 de alto risco.

Class: É a previsão final daquele ramo (Baixo Risco ou Alto Risco).

3. Análise do Padrão de Risco Identificado
Olhando para a profundidade da árvore na imagem:

Padrões Lineares vs. Não Lineares: O modelo percebe que nem toda lesão muscular é grave. Ele condiciona: "É uma lesão muscular? Se sim, qual a temporada? Se for a temporada X, o risco aumenta". 

Cores e Intensidade: As caixas coloridas (geralmente laranja para uma classe e azul para outra) facilitam a identificação visual das zonas de perigo. Se uma folha final é azul escura e tem um Gini baixo, você encontrou um perfil de jogador com altíssimo risco de lesão grave.

