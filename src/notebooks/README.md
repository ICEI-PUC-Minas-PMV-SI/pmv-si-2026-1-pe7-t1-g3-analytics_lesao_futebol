## 🚀 Como Rodar o Notebook no VS Code

Siga este roteiro passo a passo para configurar o ambiente e executar as análises:

### 1. Clonar o Repositório

Abra o terminal do seu computador e execute:

````bash
git clone https://github.com/ICEI-PUC-Minas-PMV-SI/pmv-si-2026-1-pe7-t1-g3-analytics_lesao_futebol.git
cd pmv-si-2026-1-pe7-t1-g3-analytics_lesao_futebol

### 2. Criar e Ativar um Ambiente Virtual
Abra a pasta do projeto no VS Code e, no terminal (`Ctrl + '`), execute:

**No Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
````

**No Linux ou macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

_Você verá um prefixo `(venv)` no terminal indicando que o ambiente está ativo._

### 3. Instalar as Dependências

Com o ambiente virtual ativo, instale as bibliotecas necessárias:

```bash
pip install pandas numpy matplotlib seaborn kagglehub ipykernel
```

### 4. Abrir o Notebook

1. No explorador de arquivos (lado esquerdo do VS Code), navegue até `src/notebooks/`.
2. Clique no arquivo `.ipynb` para abrir o editor de notebook.

### 5. Selecionar o Kernel

1. No canto **superior direito** do notebook, clique em **"Selecionar Kernel"** (_Select Kernel_).
2. Escolha **"Ambientes Python..."** (_Python Environments_).
3. Selecione o interpretador do ambiente virtual criado:
   - **Windows:** `.\venv\Scripts\python.exe`
   - **Linux/macOS:** `./venv/bin/python`

### 6. Executar as Células

1. Clique em **"Executar Tudo"** (_Run All_) na barra superior do notebook.
2. Ou clique no ícone de **Play** em cada célula individualmente.
3. Execute sempre **de cima para baixo**.
