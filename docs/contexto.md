# Introdução

Texto descritivo introdutório apresentando a visão geral do projeto a ser desenvolvido considerando o contexto em que ele se insere, os objetivos gerais, a justificativa e o público-alvo do projeto.

## Problema

Lesões em jogadores profissionais de futebol são eventos comuns e multifatoriais. Elas dependem de características individuais (idade, posição, condição física e histórico), do contexto competitivo (calendário, intensidade e viagens) e da forma como o atleta é exposto a jogos e treinos, o que torna o risco de lesão difícil de prever e controlar.

No cenário de clubes e ligas, antecipar situações de maior risco pode contribuir para decisões mais informadas sobre prevenção, preparação física e gestão do elenco. Assim, investigar padrões históricos de lesões e aplicar técnicas de aprendizado de máquina para apoiar a análise do risco de lesão em jogadores representa um problema relevante dentro do esporte de alto rendimento.

## Questão de pesquisa

O problema central deste estudo é aplicar técnicas de aprendizado de máquina em dados históricos das principais ligas europeias para identificar padrões e prever o risco de lesões em jogadores profissionais de futebol.

## Objetivos preliminares

Nesta seção, você deve apresentar os objetivos preliminares do trabalho, deixando claro que o objetivo geral é experimentar modelos de aprendizado de máquina adequados para solucionar o problema descrito anteriormente.

Além do objetivo geral, é importante definir pelo menos dois objetivos específicos, que direcionem a investigação de acordo com o foco que o grupo pretende adotar. Esses objetivos específicos podem envolver: 
* Explorar um determinado tipo de modelagem ou técnica de aprendizado de máquina;
* Comparar diferentes abordagens para resolver o mesmo problema;
* Aplicar o modelo em um cenário real ou simulado;
* Otimizar parâmetros para melhorar métricas específicas de desempenho.

Exemplo:
Objetivo específico 1: Predizer a tendência de alta, estabilidade ou queda de uma determinada ação em uma janela de tempo definida.
Objetivo específico 2: Estimar o valor exato da ação ao final do período analisado.

**Importante:** À medida que a pesquisa/experimentação avança, os objetivos podem ser ajustados ou refinados. Mantenha essa seção atualizada no repositório para refletir o andamento e as novas decisões do projeto.
 
> **Links Úteis**:
> - [Objetivo geral e objetivo específico: como fazer e quais verbos utilizar](https://blog.mettzer.com/diferenca-entre-objetivo-geral-e-objetivo-especifico/)

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

Nesta seção, descreva abordagens da literatura que tratam problemas semelhantes ao seu. Seu objetivo é documentar métodos, dados, métricas e resultados.

### O que levantar (mínimo 5 trabalhos)
Para **cada estudo encontrado** aderente à temática do grupo, registre de forma objetiva:
* Problema e contexto: que problema o trabalho buscou resolver e em qual domínio/cenário foi aplicado.
* Dados (dataset): origem, tamanho, período, variáveis/atributos, pré-processamentos relevantes (faltantes, balanceamento, normalização).
* Abordagem/algoritmos: algoritmos utilizados e parâmetros principais (quando informados).
* Métricas de avaliação: quais e por quê (ex.: Acurácia, F1, AUC, RMSE, MAE, etc.).
* Resultados: principais números, comparações internas, limitações citadas e conclusões.

* Texto-síntese crítico (2–4 parágrafos) respondendo:
- O que os estudos concordam? Onde divergem?
- Quais lacunas permanecem (dados, métricas, cenários, limitações técnicas/éticas)?
- Como seu projeto se alinha aos estudos identificados?

**Dica:** Prefira artigos dos últimos 5 anos ou referências clássicas indispensáveis.

### Ferramentas inteligentes permitidas
Você pode utilizar: Perplexity, SciSpace, Elicit, Research Rabbit, Litmaps.
Use-as para descoberta, organização e triagem de literatura. 

**Atenção:** 
* Sempre acesse a fonte original (PDF/artigo) antes de citar; verifique números e conclusões.
* Registre DOI/URL oficial e dados bibliográficos completos.
* Evite “alucinações” das ferramentas: desconfie de referências sem DOI ou que você não consiga localizar oficialmente.
* Use as ferramentas inteligentes para mapear redes de citação (Research Rabbit), mapas de tópicos (Litmaps), filtrar por período e gerar resumos iniciais (Perplexity/SciSpace/Elicit).
* Leia os trabalhos mais promissores e descarte estudos fora de escopo.

> **Links Úteis**:
> - [Google Scholar](https://scholar.google.com/)
> - [IEEE Xplore](https://ieeexplore.ieee.org/Xplore/home.jsp)
> - [Science Direct](https://www.sciencedirect.com/)
> - [ACM Digital Library](https://dl.acm.org/)

# Descrição do _dataset_ selecionado

## Descrição do Conjunto de Dados

O presente estudo utiliza o conjunto de dados intitulado *European Football Injuries 2020–2025*,
disponibilizado publicamente na plataforma Kaggle pelo autor Sanan Muzaffarov (2024). O dataset
consolida registros detalhados de lesões e indisponibilidades de atletas profissionais de futebol
que atuam nas cinco principais ligas europeias (Bundesliga, Premier League, La Liga, Ligue 1 e
Serie A), cobrindo o período compreendido entre as temporadas 2020/2021 e 2024/2025.

Os dados estão licenciados sob a **Creative Commons Attribution-ShareAlike 4.0 International
(CC BY-SA 4.0)**, o que permite o uso e a adaptação para fins acadêmicos, mediante a devida
atribuição de autoria e compartilhamento sob a mesma licença.

---

### Estrutura e Atributos

O arquivo analisado no Google Colab, por meio da biblioteca kagglehub e do link (`"sananmuzaffarov/european-football-injuries-2020-2025"`) é composto por **15.603 registros** e **11 atributos**, conforme detalhado no Quadro 1. Cada observação representa um evento único de afastamento de um jogador,
permitindo análises longitudinais e transversais sobre a incidência e a gravidade das lesões.

**Quadro 1 – Dicionário de Variáveis do Dataset**

| Variável               | Descrição                                          | Tipo de Dado    | Unidade / Formato          | Exemplos                          |
|------------------------|----------------------------------------------------|-----------------|----------------------------|-----------------------------------|
| `Season`               | Temporada da ocorrência                            | Categórico      | Texto                      | `20/21`, `24/25`                  |
| `Injury`               | Diagnóstico ou natureza da lesão                   | Categórico      | Texto                      | `Hamstring injury`, `Knee injury` |
| `Days`                 | Período total de afastamento informado             | Texto/Numérico  | Dias                       | `43 days`, `8 days`               |
| `Games missed`         | Quantidade de partidas oficiais perdidas           | Numérico        | Inteiro (jogos)            | `9`, `2`, `145`                   |
| `injury_from_parsed`   | Data de início da indisponibilidade                | Temporal        | MM/DD/AAAA                 | `1/28/2021`, `11/6/2020`          |
| `injury_until_parsed`  | Data de retorno às atividades                      | Temporal        | MM/DD/AAAA                 | `3/11/2021`, `11/13/2020`         |
| `player_name`          | Identificação do atleta                            | Categórico      | Texto                      | `Benjamin Pavard`                 |
| `player_age`           | Idade do atleta no momento do evento               | Numérico        | Anos (inteiro)             | `19`, `25`, `43`                  |
| `player_position`      | Posição tática principal do jogador                | Categórico      | Texto                      | `Goalkeeper`, `Centre-Back`       |
| `club`                 | Clube de vínculo do atleta                         | Categórico      | Texto                      | `Bayern Munich`                   |
| `league`               | Liga nacional correspondente ao clube              | Categórico      | Texto                      | `Bundesliga`, `Serie A`           |

---

### Qualidade e Tratamento dos Dados

A análise preliminar de qualidade revelou um conjunto de dados robusto, conforme sintetizado
no Quadro 2.

**Quadro 2 – Resumo da Qualidade dos Dados**

| Dimensão                          | Resultado                                                                 |
|-----------------------------------|---------------------------------------------------------------------------|
| Valores faltantes (*missing*)     | **0** em todas as 11 variáveis (0% *missing data*)                        |
| Duplicatas exatas                 | **0** linhas duplicadas                                                   |
| Inconsistências temporais         | **0** casos com data de retorno anterior à data de início                 |
| Coerência `Days` vs. intervalo    | **0** divergências superiores a 3 dias entre duração textual e datas      |
| Outliers — `Games missed`         | **1.367** registros acima do limite IQR (~13,5 jogos); máx. **145**       |
| Outliers — `Days` (numérico)      | **1.497** registros acima do limite IQR (~84 dias); máx. **1.013 dias**   |

Os valores extremos identificados foram mantidos na análise inicial por representarem casos
reais de lesões de longa duração — como rupturas de ligamento cruzado anterior —, típicas
do contexto do futebol profissional de alto rendimento. Recomenda-se, contudo, o uso de
medidas robustas (mediana e percentis) em análises descritivas, e a avaliação de técnicas
de *winsorização* ou segmentação por tipo de lesão em modelos preditivos.

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

