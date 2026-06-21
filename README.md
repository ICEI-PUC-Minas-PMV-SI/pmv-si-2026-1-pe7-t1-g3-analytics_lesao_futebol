# PIPELINE DE PREDIÇÃO DE LESÕES NO FUTEBOL

`CURSO: Sistemas de Informação`

`DISCIPLINA: Projeto - Pesquisa e Experimentação em Sistemas de Informação`

`SEMESTRE: 7º`

Este é um projeto que realiza a análise, extração e treinamento de IA de um conjunto de dados intitulado European Football Injuries 2020–2025, disponibilizado publicamente na plataforma Kaggle pelo autor Sanan Muzaffarov (2024).  O dataset consolida registros detalhados de lesões e indisponibilidades de atletas profissionais de futebol que atuam nas cinco principais ligas europeias (Bundesliga, Premier League, La Liga, Ligue 1 e Serie A), cobrindo o período compreendido entre as temporadas 2020/2021 e 2024/2025. 

O objetivo principal do projeto é de conseguir prever a probabilidade da ocorrência de lesões em jogadores das principais ligas européias de futebol.   


## Integrantes

* Lucas Henrique Oliveira Prado
* Beatriz Fontainha de Castro
* André Fabiano de Andrade Lima
* Felipe Paiva dos Santos
* Lucas Peres Dias Costa
* Ramir Aguiar Ribeiro Junior
* Ricardo de Andrade

## Orientador

* Neil Paiva Tizzo

---

# Planejamento

| Etapa         | Atividades |
|  :----:   | ----------- |
| ETAPA 1         |[Documentação de Contexto e levantamento dos dados](docs/contexto.md) <br> |
| ETAPA 2         |[Conhecendo os dados](docs/conhecendo-dados.md) <br> |
| ETAPA 3         |[Preparação dos dados, construção e avaliação do modelo proposto](docs/construindo-modelo.md) |
| ETAPA 4         |[Preparação dos dados, construção, avaliação e comparação dos modelos propostos](docs/construindo-modelos.md) |
| ETAPA 5         |[Implantação e apresentação da solução](docs/implantação-apresentacao.md) <br>  |

---

# 🚀 Instruções de Utilização da API de Produção

Este repositório contém o pipeline de produção modularizado para predição do tempo de recuperação de lesões (em dias) utilizando Machine Learning. Siga os passos abaixo para executar a aplicação localmente utilizando o gateway de produção.

#### 📋 Pré-requisitos
Certifique-se de ter o Python 3.10 ou 3.11 instalado em sua máquina.

## 🔧 Instalação e Configuração

1. **Clonar o Repositório:**
   
   ```bash
   git clone [https://github.com/pucminas/seu-repositorio.git](https://github.com/pucminas/seu-repositorio.git)
   cd seu-repositorio

2. **Criar e Ativar Ambiente Virtual (Recomendado):**

   ```bash
   python -m venv venv
   
  ### No Windows:
     venv\Scripts\activate
  
  ### No Linux/Mac:
     source venv/bin/activate

3. **Rodando a Aplicação:**
   
   ```bash
   waitress-serve --listen=0.0.0.0:5000 wsgi:app

---

# Código

<li><a href="src/injury_prediction_advanced"> Código Fonte</a></li>

# Apresentação

<li><a href="presentation/README.md"> Apresentação da solução</a></li>
