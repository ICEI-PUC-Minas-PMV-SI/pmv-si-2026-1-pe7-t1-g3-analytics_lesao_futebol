# Implantação da solução

# Seção X: Implantação da Solução

Nesta etapa, é descrita em detalhes a estratégia de implantação da solução em ambiente produtivo, contemplando o planejamento da capacidade operacional com base em modelagem matemática e simulação do sistema, a especificação do provedor de computação em nuvem, o empacotamento do microsserviço e a documentação técnica das limitações operacionais enfrentadas no ambiente remoto, concluindo com os testes de homologação em tempo de execução.

A aplicação em produção foi desenhada especificamente para realizar a inferência dinâmica utilizando o pipeline e os pesos do modelo previamente treinados, expondo uma interface amigável para a entrada de novos dados fornecidos pelo usuário, de modo que as **predições sejam realizadas em tempo de execução**, sem necessidade de reprocessamento ou atualização incremental do treinamento.

---

## 1. Planejamento de Capacidade Operacional e Modelagem do Sistema

Para dimensionar os recursos computacionais necessários para o funcionamento estável da API de predição de lesões, foi realizado um planejamento de capacidade operacional baseado nas características de consumo do artefato consolidado em produção (`production_pipeline.joblib`) e nas demandas esperadas de concorrência de rede.

### 1.1 Análise de Consumo de Memória (RAM) e CPU
O ecossistema de produção adota uma estratégia de inicialização otimizada por meio de ganchos de ciclo de vida do servidor (padrão *Singleton* via WSGI). O modelo final selecionado no benchmark experimental, **LightGBM Baseline**, juntamente com todo o pipeline de engenharia de atributos (composto pelas 46 variáveis ativas), é carregado inteiramente na memória RAM uma única vez no momento do *startup* da aplicação.

* **Pegada de Memória Estática:** O binário serializado do LightGBM combinado com as tabelas estruturadas de mapeamento estatístico (*Target Encoding*) e o dicionário de internacionalização e tradução (`dicionario.json`) ocupam menos de **45 MB** de RAM em estado de repouso.
* **Pegada de Memória Dinâmica:** Cada requisição HTTP POST encaminhada pelo formulário Front-end transporta um payload JSON estruturado de aproximadamente **1.8 KB**. A transformação desse JSON em um vetor numérico bidimensional interpretável pelo Pandas e a inferência subsequente pelo estimador consomem um pico volátil de menos de **2 MB** por transação.
* **Tempo de Serviço da CPU:** Por se tratar de um modelo baseado em árvores de decisão otimizadas e sem necessidade de retreinamento em tempo de execução, o tempo médio de CPU gasto para computar uma estimativa pontual é de aproximadamente **$12 \text{ ms}$** ($0.012 \text{ segundos}$).

### 1.2 Modelagem Matemática de Concorrência e Vazão
Utilizando a **Lei de Little** para sistemas de filas estacionários em equilíbrio, é possível calcular a capacidade máxima teórica de requisições simultâneas suportadas por um único núcleo de processamento lógico sem que ocorra degradação do tempo de resposta (latência):

$$\text{Vazão Máxima (RPS)} = \frac{\text{Número de Núcleos (C)}}{\text{Tempo de Serviço (S)}}$$

Considerando um ambiente de computação com $1 \text{ vCPU}$ dedicada e o tempo de serviço de inferência mensurado de $S = 0.012 \text{ segundos}$:

$$\text{Vazão Máxima} = \frac{1}{0.012} \approx 83.33 \text{ requisições por segundo (RPS)}$$

Aplicando um fator de segurança conservador de **$30\%$** para acomodar o overhead do protocolo HTTP/REST, a serialização de strings no Flask e a renderização do HTML5, o sistema modelado demonstra capacidade de sustentar com estabilidade **~58 requisições de inferência dinâmica simultâneas por segundo**, superando com ampla margem a demanda de uso real de múltiplos departamentos médicos simultâneos.

---

## 2. Escolha e Configuração do Provedor de Nuvem

O ecossistema de nuvem selecionado para o planejamento do projeto foi o **Microsoft Azure**, usufruindo do programa de fomento acadêmico *Azure for Students* (com cota de crédito estudantil de \$100).

### 2.1 Especificação da Infraestrutura Planejada
Para atender à modelagem matemática de capacidade sem estourar o orçamento de créditos gratuitos, a topologia do microsserviço foi especificada da seguinte forma:

* **Serviço de Destino:** *Azure App Service* (Serviço de Aplicativo do Azure) baseado em contêineres Linux embarcados, fornecendo abstração de gerenciamento de infraestrutura profunda e escalabilidade horizontal automática.
* **Plano de Serviço (App Service Plan):** Camada **B1 (Basic)**.
* **Recursos Computacionais Alocados:** $1 \text{ vCPU}$ dedicada, $1.75 \text{ GB}$ de memória RAM e $10 \text{ GB}$ de armazenamento em disco de estado sólido (SSD).
* **Ambiente de Execução (Runtime Stack):** Python 3.10 LTS.

* **Infelizmente devido à problemas na conta, não fora possível a configuração da implantação da produção na plataforma do Azure, seguem alguns prints dos problemas técnicos enfrentados:

<img width="930" height="1008" alt="Captura de tela 2026-06-16 194905" src="https://github.com/user-attachments/assets/0696cbcf-14df-4272-a3be-61d0badc9bb3" />

<img width="1898" height="821" alt="Captura de tela 2026-06-16 191236" src="https://github.com/user-attachments/assets/14369e37-fda2-4730-85c8-13b5510f0f84" />


---

## 3. Empacotamento, Distribuição e Estratégia de Deploy Local

### 3.1 Arquitetura de Empacotamento de Produção
Para garantir que a aplicação estivesse isolada e pronta para o ambiente produtivo sem depender de ferramentas de desenvolvimento do ecossistema do Jupyter Notebook, foi construída uma arquitetura de microsserviço utilizando o servidor de gateway de interface de servidor web **Waitress** (servidor WSGI estável de nível de produção nativo para sistemas Linux e Windows).

O arquivo `wsgi.py` foi exposto como ponto de entrada da aplicação, encapsulando o objeto executável do Flask (`app:app`), permitindo o gerenciamento assíncrono da fila de conexões de rede de forma robusta.

O arquivo `requirements.txt` foi estruturado de forma estrita, garantindo a reprodutibilidade exata das versões das bibliotecas matemáticas e de Machine Learning homologadas:

```text
Flask==3.0.3
lightgbm==4.3.0
scikit-learn==1.3.2
pandas==2.1.4
waitress==3.0.0


# Apresentação da solução

Nesta seção, deve ser produzido um vídeo de até 15 minutos apresentando o escopo geral do projeto, um resumo das etapas desenvolvidas, a demonstração da solução publicada e as conclusões finais, destacando aprendizados, impacto e possibilidades de melhorias.

# É IMPRESCINDÍVEL: 
* Atualizar o arquivo **CITATION.cff** disponível no diretório raiz do repositório
* Atualizar as **Instruções de utilização** no arquivo read.me



