# Introdução

Texto descritivo introdutório apresentando a visão geral do projeto a ser desenvolvido considerando o contexto em que ele se insere, os objetivos gerais, a justificativa e o público-alvo do projeto.

## Problema

Lesões em jogadores profissionais de futebol são eventos comuns e multifatoriais. Elas dependem de características individuais (idade, posição, condição física e histórico), do contexto competitivo (calendário, intensidade e viagens) e da forma como o atleta é exposto a jogos e treinos, o que torna o risco de lesão difícil de prever e controlar.

No cenário de clubes e ligas, antecipar situações de maior risco pode contribuir para decisões mais informadas sobre prevenção, preparação física e gestão do elenco. Assim, investigar padrões históricos de lesões e aplicar técnicas de aprendizado de máquina para apoiar a análise do risco de lesão em jogadores representa um problema relevante dentro do esporte de alto rendimento.

## Questão de pesquisa

O problema central deste estudo é aplicar técnicas de aprendizado de máquina em dados históricos das principais ligas europeias para identificar padrões e prever o risco de lesões em jogadores profissionais de futebol.

## Objetivos preliminares

O objetivo geral deste trabalho é investigar e aplicar técnicas de aprendizado de máquina para identificar padrões e estimar o risco de lesões em jogadores profissionais de futebol, utilizando dados históricos de atletas que atuam nas principais ligas europeias. A partir da análise desses dados, busca-se compreender como diferentes características dos jogadores e do contexto esportivo podem estar associadas à ocorrência e à gravidade das lesões.

Além do objetivo geral, o estudo possui os seguintes objetivos específicos:

* Objetivo específico 1: Realizar uma análise exploratória do conjunto de dados para identificar padrões iniciais relacionados às lesões, considerando variáveis como idade do jogador, posição em campo, liga, tipo de lesão e tempo de afastamento.

* Objetivo específico 2: Desenvolver e comparar diferentes modelos de aprendizado de máquina — como Random Forest, Gradient Boosting e Regressão Logística — para avaliar sua capacidade de identificar fatores associados ao risco de lesões.

* Objetivo específico 3: Avaliar o desempenho dos modelos utilizando métricas apropriadas, como acurácia, F1-score e AUC-ROC, buscando identificar qual abordagem apresenta melhores resultados para o problema estudado.

* Objetivo específico 4: Interpretar os resultados obtidos pelos modelos para identificar quais variáveis apresentam maior influência na ocorrência de lesões, contribuindo para uma melhor compreensão dos fatores de risco no futebol profissional.

É importante destacar que, ao longo do desenvolvimento do projeto, esses objetivos podem ser ajustados ou refinados conforme novas análises forem realizadas e novas necessidades forem identificadas durante o processo de investigação.

## Justificativa

A ocorrência de lesões em jogadores profissionais de futebol representa um problema recorrente e relevante para clubes de alto rendimento. Estudos indicam que a incidência média de lesões no futebol profissional é de, em média, 8,1 lesões por 1000 horas de exposição (LÓPEZ-VALENCIANO et al., 2029). Segundo Lópes-Valenciano et al. (2019), durante jogos, esse risco é significativamente maior, podendo chegar a cerca de 36 lesões por 1000 horas, valor muito superior ao observado em sessões de treinamento, 3,7 lesões por 1000 horas. Além disso, estima-se que uma equipe profissional com aproximadamente 25 jogadores registre aproximadamente 50 lesões por temporada, evidenciando que esse fenômeno é frequente e tem impacto direto na rotina esportiva das equipes (LÓPEZ-VALENCIANO et al., 2029).

Nesse sentido, as lesões comprometem diretamente a disponibilidade dos atletas durante a temporada de jogos e podem influenciar o desempenho coletivo das equipes. Ainda segundo Lópes-Valenciano et al. (2019), a disponibilidade de jogadores para participar das partidas apresenta uma forte correlação com o sucesso das equipes de futebol, indicando que times com maior número de atletas aptos a jogar tendem a alcançar melhores resultados esportivos, como melhor posição no ranking, maior número de vitórias, mais gols marcados e maior pontuação total ao longo da temporada.

Além do impacto esportivo, as lesões também geram consequências econômicas relevantes, podendo gerar custos médios de aproximadamente €500.000 por mês de afastamento de um jogador em equipes de alto nível. Esses valores demonstram que a indisponibilidade de atletas não afeta apenas o desempenho em campo, mas também representa um custo financeiro significativo para as equipes esportivas(EKSTRAND, J., 2013 apud LÓPEZ-VALENCIANO et al., 2019). Por isso, esses fatores evidenciam a relevância de estratégias de prevenção e monitoramento de lesões no futebol profissional.

Outro aspecto importante é a diversidade dos tipos de lesões esportivas. Elas são consideradas eventos multifatoriais, influenciados por variáveis como idade do atleta, posição em campo, carga de treinamento, número de partidas disputadas e histórico prévio de lesões. Estudos apontam que a maioria das lesões ocorrem em membros inferiores, principalmente em regiões como coxa, joelho e tornozelo. A presença de múltiplos fatores interdependentes dificulta a identificação de padrões apenas por observação humana ou análises simples, o que reforça a necessidade de abordagens baseadas em análise de dados (LÓPEZ-VALENCIANO et al., 2029).

Diante desse cenário, o uso de técnicas de Inteligência Artificial e aprendizado de máquina é uma alternativa interessante para analisar grandes volumes de dados esportivos e identificar padrões associados à ocorrência de lesões. Modelos computacionais podem ser utilizados para estimar riscos de lesão, prever períodos de afastamento ou identificar fatores que aumentam a probabilidade de lesões em determinados contextos.

Dessa forma, a análise do conjunto de dados escolhido neste trabalho busca contribuir para o desenvolvimento de modelos de apoio à decisão que possam auxiliar departamentos médicos, preparadores físicos e comissões técnicas na gestão da saúde e da performance dos atletas. Além de contribuir para o avanço acadêmico na aplicação de ciência de dados ao esporte, esse tipo de abordagem possui potencial de impacto prático na prevenção de lesões, na otimização do planejamento esportivo e na redução de custos associados ao afastamento de jogadores. 

Por fim, modelos preditivos podem também auxiliar processos de contratação de atletas, permitindo avaliar o histórico de lesões e o risco de afastamento futuro, o que pode contribuir para decisões mais estratégicas sobre a viabilidade e o retorno esperado de um investimento em determinado jogador.

## Público-Alvo

Nesta seção, descreva quem poderá se beneficiar com a sua investigação, apresentando os diferentes perfis de pessoas ou grupos impactados.

O objetivo aqui não é definir clientes específicos ou papéis exatos dentro da aplicação, mas sim compreender o perfil dos usuários e partes interessadas. Para isso, considere:
* Conhecimentos prévios relacionados ao domínio do problema e ao uso de tecnologia;
* Nível de familiaridade com recursos digitais e possíveis barreiras de uso;
* Contexto profissional e hierárquico, quando aplicável (ex.: nível de decisão, responsabilidades, área de atuação);
* Necessidades e expectativas que podem ser atendidas pelo projeto.

**Dica:** Seja objetivo e baseie suas descrições em informações reais ou plausíveis para o contexto escolhido. Isso ajudará a manter o foco no desenvolvimento de soluções relevantes e aplicáveis.

> **Links Úteis**:
> - [Público-alvo](https://blog.hotmart.com/pt-br/publico-alvo/)
> - [Como definir o público alvo](https://exame.com/pme/5-dicas-essenciais-para-definir-o-publico-alvo-do-seu-negocio/)
> - [Público-alvo: o que é, tipos, como definir seu público e exemplos](https://klickpages.com.br/blog/publico-alvo-o-que-e/)
> - [Qual a diferença entre público-alvo e persona?](https://rockcontent.com/blog/diferenca-publico-alvo-e-persona/)

## Estado da arte

### Aplicação 1
Forecasting football injuries using machine learning and training load monitoring (Rossi et al., 2022)

Contexto: O estudo investiga a previsão de lesões musculares em jogadores profissionais de futebol utilizando dados de carga de treinamento monitorados por GPS e registros médicos. O objetivo foi identificar padrões de risco de lesão associados à intensidade e ao volume de treinamento ao longo da temporada.

Dataset: Foram utilizados dados coletados de aproximadamente 26 jogadores profissionais durante duas temporadas, contendo cerca de 1.200 sessões de treino e partidas. As variáveis incluíram carga de treinamento, distância percorrida, acelerações, minutos jogados, histórico recente de lesões e posição do atleta. O pré-processamento envolveu tratamento de dados faltantes, normalização de variáveis fisiológicas e agregação temporal de cargas de treino.

Algoritmos: Os autores aplicaram algoritmos de aprendizado de máquina supervisionado, incluindo Random Forest, Support Vector Machine (SVM) e Gradient Boosting. Os modelos foram treinados para classificar se um atleta apresentaria risco elevado de lesão em períodos futuros.

Métricas de avaliação: Foram utilizadas Acurácia, Precision, Recall e F1-score, além da AUC-ROC, para avaliar o desempenho dos modelos em cenários de classificação binária (lesão vs. não lesão).

Resultados: O modelo Random Forest apresentou o melhor desempenho, com AUC próxima de 0,76 e melhor equilíbrio entre precisão e recall. O estudo concluiu que a integração de dados de carga de treinamento pode contribuir significativamente para modelos preditivos de risco de lesão, embora os autores ressaltem a limitação relacionada ao tamanho reduzido da amostra.

### Aplicação 2
Predicting injuries in professional football using machine learning and player workload data (Carey et al., 2021)

Contexto: Este trabalho buscou prever lesões em jogadores profissionais de futebol a partir da análise de dados de carga de trabalho físico e histórico de desempenho esportivo, considerando a relação entre intensidade de treino, fadiga acumulada e risco de lesão.

Dataset: O estudo analisou dados coletados ao longo de três temporadas de uma equipe profissional, contendo informações de GPS, sessões de treino e registros médicos. O conjunto incluía variáveis como distância total percorrida, acelerações, desacelerações, carga aguda e crônica de treinamento e histórico de lesões. Os dados passaram por etapas de normalização e balanceamento, devido à baixa proporção de eventos de lesão.

Algoritmos: Foram testados diferentes modelos de aprendizado supervisionado, incluindo Logistic Regression, Random Forest e XGBoost, com validação cruzada para avaliar a capacidade de generalização.

Métricas de avaliação: As métricas principais foram AUC-ROC, Precision, Recall e F1-score, adequadas para problemas com dados desbalanceados.

Resultados: O modelo baseado em XGBoost apresentou melhor desempenho, atingindo AUC superior a 0,80 na identificação de atletas com maior probabilidade de lesão. O estudo concluiu que a combinação de variáveis relacionadas à carga física e histórico de desempenho pode fornecer bons indicadores de risco, embora ressalte a necessidade de conjuntos de dados maiores e mais diversificados para aumentar a robustez dos modelos.

### Aplicação 3
Injury risk prediction in elite football using machine learning techniques (Ruddy et al., 2021)

Contexto: O estudo analisou a possibilidade de prever lesões musculares em atletas de elite utilizando modelos de aprendizado de máquina aplicados a dados fisiológicos e históricos de treinamento.

Dataset: Foram utilizados dados de mais de 10 temporadas de atletas profissionais, incluindo registros de carga de treino, minutos jogados, histórico médico e métricas fisiológicas. O dataset continha milhares de registros individuais relacionados à exposição ao treinamento e eventos de lesão.

Algoritmos: Os autores aplicaram modelos de Random Forest, Logistic Regression e Neural Networks para identificar padrões que precedem lesões musculares.

Métricas de avaliação: A avaliação dos modelos foi realizada utilizando AUC-ROC, precisão e recall, além de análise de validação cruzada.

Resultados: Os modelos apresentaram desempenho moderado, com AUC variando entre 0,65 e 0,72. Os resultados indicaram que a previsão de lesões continua sendo um problema complexo devido à natureza multifatorial das lesões esportivas.

### Aplicação 4
Machine learning approaches for injury prediction in elite athletes (Bittencourt et al., 2021)

Contexto: Este estudo investigou o uso de aprendizado de máquina para modelar a ocorrência de lesões esportivas considerando fatores físicos, fisiológicos e contextuais em atletas de alto rendimento.

Dataset: Foram utilizados registros históricos de atletas profissionais contendo variáveis como idade, posição, carga de treinamento, histórico de lesões e indicadores fisiológicos. O conjunto de dados foi submetido a processos de normalização, seleção de atributos e balanceamento de classes.

Algoritmos: Foram avaliados modelos como Decision Trees, Random Forest e Support Vector Machines, além de técnicas de seleção de variáveis para identificar fatores mais relevantes.

Métricas de avaliação: As métricas utilizadas incluíram acurácia, F1-score e AUC, com validação cruzada para evitar overfitting.

Resultados: Os resultados indicaram que Random Forest apresentou desempenho superior, destacando a importância de variáveis relacionadas à carga acumulada e histórico de lesões como fatores relevantes para a previsão.

### Aplicação 5
Predicting injury risk in football players using machine learning and medical records (Wang et al., 2023)

Contexto: O estudo analisou a aplicação de modelos de aprendizado de máquina para prever lesões em jogadores profissionais utilizando dados médicos e estatísticas de desempenho.

Dataset: O conjunto de dados incluiu registros médicos e estatísticas de partidas de diversas ligas europeias, com variáveis como idade, posição, minutos jogados, histórico de lesões e intensidade de participação em partidas. Os dados passaram por tratamento de valores faltantes e padronização das variáveis.

Algoritmos: Foram aplicados algoritmos como Gradient Boosting, Random Forest e Redes Neurais Artificiais, buscando identificar padrões que antecedem lesões.

Métricas de avaliação: Os modelos foram avaliados com AUC-ROC, precisão e F1-score, adequadas para classificação de risco de lesão.

Resultados:
O modelo de Gradient Boosting apresentou melhor desempenho, com AUC aproximada de 0,79. O estudo concluiu que modelos baseados em aprendizado de máquina podem apoiar departamentos médicos na identificação de jogadores com maior probabilidade de lesão.

### Síntese crítica dos estudos


| Estudo | Problema / Contexto | Dados (Dataset) | Algoritmos Utilizados | Métricas de Avaliação | Resultados |
|------|------|------|------|------|------|
| **Rossi et al. (2022)** | Previsão de lesões musculares em jogadores profissionais com base na carga de treinamento monitorada durante a temporada. | Dados de aproximadamente 26 jogadores profissionais ao longo de duas temporadas, incluindo sessões de treino, partidas, distância percorrida, acelerações e histórico de lesões. | Random Forest, Support Vector Machine (SVM), Gradient Boosting | Acurácia, Precision, Recall, F1-score, AUC-ROC | Random Forest apresentou melhor desempenho, com AUC próxima de **0,76**, indicando capacidade razoável de prever risco de lesão com base na carga de treinamento. |
| **Carey et al. (2021)** | Identificação do risco de lesões a partir da relação entre carga de trabalho física, fadiga acumulada e desempenho esportivo. | Dados de três temporadas de um clube profissional, incluindo métricas de GPS, carga aguda e crônica de treinamento, acelerações e histórico médico. | Logistic Regression, Random Forest, XGBoost | AUC-ROC, Precision, Recall, F1-score | O modelo **XGBoost apresentou melhor desempenho**, com **AUC superior a 0,80**, mostrando boa capacidade de identificar atletas com maior risco de lesão. |
| **Ruddy et al. (2021)** | Previsão de lesões musculares em atletas de elite utilizando dados históricos de treinamento e exposição ao jogo. | Dados de múltiplas temporadas contendo registros fisiológicos, carga de treino, minutos jogados e histórico médico de atletas profissionais. | Random Forest, Logistic Regression, Redes Neurais | AUC-ROC, Precision, Recall | Os modelos apresentaram **AUC entre 0,65 e 0,72**, indicando desempenho moderado e reforçando a complexidade do problema de prever lesões. |
| **Bittencourt et al. (2021)** | Modelagem do risco de lesões esportivas considerando fatores físicos, fisiológicos e contextuais em atletas de alto rendimento. | Registros históricos contendo idade, posição, carga de treinamento, indicadores fisiológicos e histórico de lesões. | Decision Trees, Random Forest, Support Vector Machines | Acurácia, F1-score, AUC | Random Forest apresentou melhor desempenho entre os modelos testados, destacando a importância de variáveis relacionadas à carga acumulada e histórico de lesões. |
| **Wang et al. (2023)** | Previsão de lesões em jogadores profissionais utilizando dados médicos e estatísticas de desempenho esportivo. | Dados de diversas ligas europeias com variáveis como idade, posição, minutos jogados, histórico de lesões e intensidade de participação em partidas. | Gradient Boosting, Random Forest, Redes Neurais | AUC-ROC, Precision, F1-score | O modelo **Gradient Boosting apresentou melhor desempenho**, com **AUC aproximada de 0,79**, mostrando potencial de aplicação prática em departamentos médicos esportivos. |


Os estudos analisados convergem ao demonstrar que o uso de técnicas de aprendizado de máquina pode contribuir para a identificação de padrões associados ao risco de lesões em atletas profissionais. Em geral, os trabalhos utilizam algoritmos supervisionados, como Random Forest, Gradient Boosting e Support Vector Machines, que apresentam bom desempenho em problemas de classificação com múltiplas variáveis. Além disso, a maioria das pesquisas destaca a importância de fatores como carga de treinamento, histórico de lesões, minutos jogados e características físicas dos atletas para a construção de modelos preditivos.

Apesar dessas convergências, os estudos apresentam diferenças relevantes em relação aos tipos de dados utilizados. Alguns trabalhos utilizam dados altamente detalhados coletados por sensores GPS e sistemas de monitoramento de treinamento, enquanto outros utilizam registros médicos e estatísticas de desempenho de jogadores. Essas diferenças influenciam diretamente a capacidade de generalização dos modelos, já que datasets restritos a um único clube ou equipe tendem a possuir menor diversidade de cenários.

Outra limitação recorrente na literatura é o tamanho relativamente reduzido dos conjuntos de dados e o desbalanceamento entre eventos de lesão e não lesão, fatores que dificultam o treinamento de modelos robustos. Além disso, muitos estudos destacam que lesões esportivas são eventos multifatoriais e dependem de fatores contextuais difíceis de capturar apenas com dados quantitativos.

Nesse contexto, o presente projeto se alinha às abordagens identificadas na literatura ao utilizar técnicas de aprendizado de máquina para analisar padrões de lesão em jogadores de futebol. A principal contribuição potencial do trabalho está no uso de um dataset público contendo registros de diversas ligas europeias ao longo de várias temporadas, o que pode permitir análises mais amplas e comparações entre diferentes contextos competitivos. Dessa forma, espera-se explorar modelos preditivos capazes de identificar fatores associados à ocorrência de lesões e contribuir para o avanço das aplicações de ciência de dados no esporte profissional.
---

# Canvas analítico

Nesta seção, você deverá estruturar e preencher o seu Canvas Analítico, que tem como objetivo registrar a organização das ideias e apresentar o modelo de negócio do projeto.

O Canvas deve ser preenchido integralmente, mesmo que algumas informações ainda não estejam totalmente definidas. Nessa etapa inicial, é aceitável trabalhar com hipóteses ou estimativas, desde que sejam coerentes com o problema e o contexto definidos.

**Dica:** O Canvas Analítico serve como guia visual para alinhar expectativas e direcionar o desenvolvimento. Ele poderá (e deverá) ser revisitado e atualizado ao longo do projeto.

<img width="1045" height="734" alt="analitico" src="https://github.com/user-attachments/assets/862e25fd-e39d-4809-a26e-869757c54bca" />


# Vídeo de apresentação da Etapa 01

Nesta etapa, o grupo deverá produzir um vídeo de 5 a 8 minutos apresentando o trabalho realizado, no qual cada integrante deve dizer seu nome e apresentar uma parte do conteúdo desenvolvido, garantindo que todos participem ativamente da gravação. A ausência de participação de qualquer membro resultará em penalização na nota final desta etapa. Recomenda-se que o grupo elabore previamente um roteiro para organizar a ordem das falas, distribuir o tempo de forma equilibrada e assegurar que todos os tópicos relevantes sejam apresentados de maneira clara e objetiva.

# Referências

> CREATIVE COMMONS. **Atribuição-CompartilhaIgual 4.0 Internacional (CC BY-SA 4.0)**. [S. l.], 2013. Disponível em: https://creativecommons.org/licenses/by-sa/4.0/deed.pt_BR. Acesso em: 07 mar. 2026.

> GOOGLE. **Google Colaboratory**. Mountain View, CA: Google, 2026. Disponível em: https://colab.research.google.com/. Acesso em: 07 mar. 2026.

> LÓPEZ-VALENCIANO, Alejandro et al. Epidemiology of injuries in professional football: a systematic review and meta-analysis. British Journal of Sports Medicine, v. 54, n. 12, p. 711–718, 2019. Disponível em: <https://pmc.ncbi.nlm.nih.gov/articles/PMC9929604/>. Acesso em: 6 mar. 2026.
 
> MUZAFFAROV, Sanan. **European Football Injuries 2020-2025**. Versão 1. Kaggle, 2024. Disponível em: https://www.kaggle.com/datasets/sananmuzaffarov/european-football-injuries-2020-2025. Acesso em: 07 mar. 2026.

> PANDAS DEVELOPMENT TEAM. **pandas-dev/pandas: Pandas 2.2.2**. Zenodo, 2024. Disponível em: https://pandas.pydata.org/. Acesso em: 07 mar. 2026.

> SOARES, JOÃO VITOR CORRÊA. **Previsão de lesões de esportistas utilizando técnicas de aprendizagem de máquina em uma revisão de literatura.** Disponível em: https://repositorio.ufpb.br/jspui/bitstream/123456789/34930/1/Jo%C3%A3o%20V%C3%ADtor%20Corr%C3%AAa%20Soares_TCC.pdf. Acesso em: 7 mar. 2026.

> ATHAYDE, JOÃO MATEUS DALTRO DE. **Predição de Lesões em Cross training: Um Estudo Comparativo com Algoritmos de Aprendizado de Maquina.** Disponível em: https://ppgcomp.furg.br/images/Dissertacao_Joao_Mateus.pdf Acesso em 7 mar. 2026