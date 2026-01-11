# Tech Challenge - Fase 1

## Sistema Inteligente de Suporte ao Diagnóstico Médico

Este projeto implementa modelos de machine learning para **classificação de exames médicos**, utilizando dados estruturados e imagens para auxiliar profissionais de saúde na tomada de decisão clínica.

> ⚠️ **IMPORTANTE**: Este sistema não substitui o médico. Ele atua como ferramenta de apoio e triagem. A decisão final sempre deve ser do profissional médico qualificado.

---

## 📑 Índice

1. [Problema Abordado](#-problema-abordado)
2. [Datasets Utilizados](#-datasets-utilizados)
3. [Estrutura do Projeto](#-estrutura-do-projeto)
4. [Instalação e Configuração](#-instalação-e-configuração)
5. [📚 Guia Passo a Passo Completo](#-guia-passo-a-passo-completo)
   - [Notebook 01: Exploração de Dados Tabulares](#notebook-01-exploração-de-dados-tabulares)
   - [Notebook 02: Modelagem de Dados Tabulares](#notebook-02-modelagem-de-dados-tabulares)
   - [Notebook 03: Exploração de Imagens de Pneumonia](#notebook-03-exploração-de-imagens-de-pneumonia)
   - [Notebook 04: Modelagem CNN para Pneumonia](#notebook-04-modelagem-cnn-para-pneumonia)
   - [Notebook 05: Exploração de Mamografias](#notebook-05-exploração-de-mamografias)
   - [Notebook 06: Modelagem CNN para Câncer de Mama](#notebook-06-modelagem-cnn-para-câncer-de-mama)
6. [🔬 Detalhes Técnicos](#-detalhes-técnicos)
7. [📈 Resultados Esperados](#-resultados-esperados)
8. [🔍 Interpretabilidade](#-interpretabilidade)
9. [⚠️ Limitações e Considerações](#️-limitações-e-considerações)
10. [🐳 Docker](#-docker)
11. [📚 Documentação Adicional](#-documentação-adicional)

---

## 📌 Problema Abordado

Este projeto aborda dois tipos de classificação médica:

### 1. Classificação de Câncer de Mama (Dados Tabulares)

Classificação binária para diagnóstico de **câncer de mama** em duas categorias:

- **B (Benigno)**: Tumor benigno
- **M (Maligno)**: Tumor maligno

O modelo utiliza características clínicas numéricas obtidas de exames médicos (raio, textura, perímetro, área, suavidade, compactação, concavidade, etc.) para fazer predições.

### 2. Classificação de Imagens Médicas (CNNs)

#### 2.1 Pneumonia em Raio-X

Classificação binária de imagens de raio-X de tórax:

- **Normal**: Sem sinais de pneumonia
- **Pneumonia**: Com sinais de pneumonia

#### 2.2 Câncer de Mama em Mamografias

Classificação binária de imagens de mamografia:

- **Benigno**: Lesões benignas
- **Maligno**: Lesões malignas (câncer)

---

## 🧪 Datasets Utilizados

### Dados Tabulares

- **Dataset**: Wisconsin Breast Cancer Dataset
- **Fonte**: UCI Machine Learning Repository
- **Tamanho**: 569 amostras
- **Features**: 30 características numéricas
- **Distribuição**: ~62% benigno, ~38% maligno
- **Localização**: `data/tabular/breast-cancer.csv`

### Dados de Imagens

#### Pneumonia em Raio-X

- **Dataset**: Chest X-Ray Images (Pneumonia)
- **Fonte**: Kaggle (paultimothymooney/chest-xray-pneumonia)
- **Tipo**: Imagens de raio-X de tórax
- **Classes**: Normal, Pneumonia
- **Download**: Automático via kagglehub

#### Câncer de Mama (CBIS-DDSM)

- **Dataset**: CBIS-DDSM (Curated Breast Imaging Subset of DDSM)
- **Fonte**: Kaggle (awsaf49/cbis-ddsm-breast-cancer-image-dataset)
- **Tipo**: Imagens de mamografia
- **Classes**: Benigno, Maligno
- **Download**: Automático via kagglehub

### Características do Dataset Tabular

O dataset contém medidas computadas a partir de imagens digitalizadas de aspirados por agulha fina (FNA) de massas mamárias. As features descrevem características do núcleo celular, incluindo:

- **Raio**: Média das distâncias do centro aos pontos do perímetro
- **Textura**: Desvio padrão dos valores de escala de cinza
- **Perímetro**: Perímetro do núcleo
- **Área**: Área do núcleo
- **Suavidade**: Variação local nos comprimentos dos raios
- **Compactação**: Perímetro² / área - 1.0
- **Concavidade**: Severidade das porções côncavas do contorno
- **Pontos côncavos**: Número de porções côncavas do contorno
- **Simetria**: Medida de simetria
- **Dimensão fractal**: Aproximação "coastline" - 1

Cada feature possui três versões: `_mean` (média), `_se` (erro padrão), `_worst` (pior valor).

---

## 🏗 Estrutura do Projeto

```
tech-challenge-8IADT/
├── data/
│   ├── tabular/
│   │   └── breast-cancer.csv          # Dataset tabular
│   └── images/
│       ├── pneumonia/                  # Dataset de pneumonia (baixado)
│       └── breast_cancer/             # Dataset de câncer de mama (baixado)
├── notebooks/
│   ├── 01_tabular_exploracao.ipynb   # EDA dados tabulares
│   ├── 02_tabular_modelagem.ipynb    # Modelagem dados tabulares
│   ├── 03_vision_pneumonia_exploracao.ipynb   # EDA pneumonia
│   ├── 04_vision_pneumonia_modelagem.ipynb    # CNN pneumonia
│   ├── 05_vision_breast_exploracao.ipynb      # EDA câncer de mama
│   └── 06_vision_breast_modelagem.ipynb       # CNN câncer de mama
├── src/
│   ├── tabular/
│   │   ├── processing.py              # Pré-processamento tabular
│   │   └── evaluate.py                # Avaliação tabular
│   └── vision/
│       ├── data_loader.py             # Carregamento de imagens
│       ├── preprocessing.py           # Pré-processamento de imagens
│       ├── models.py                  # Arquiteturas CNN
│       └── evaluation.py              # Avaliação e Grad-CAM
├── models/
│   ├── maternal_risk_model.pkl       # Modelo tabular
│   ├── pneumonia_cnn_model.h5        # CNN pneumonia
│   └── breast_cancer_cnn_model.h5     # CNN câncer de mama
├── config.yaml                        # Configurações
├── requirements.txt                   # Dependências
├── Dockerfile                         # Containerização
├── README.md                          # Este arquivo
└── relatorio_tecnico.md               # Relatório técnico
```

---

## 🚀 Instalação e Configuração

### Pré-requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)
- Jupyter Notebook ou JupyterLab (para executar os notebooks)

### Passo 1: Clonar o Repositório

```bash
git clone <url-do-repositorio>
cd tech-challenge-8IADT
```

### Passo 2: Instalar Dependências

```bash
pip3 install -r requirements.txt
```

**Nota**: Se você receber um erro "command not found: pip", use `pip3` em vez de `pip`. No macOS, o comando geralmente é `pip3`.

**Principais dependências**:

- `pandas`: Manipulação de dados
- `numpy`: Computação numérica
- `scikit-learn`: Machine learning
- `tensorflow`: Deep learning e CNNs
- `matplotlib` e `seaborn`: Visualização
- `shap`: Interpretabilidade de modelos
- `kagglehub`: Download de datasets do Kaggle
- `pillow`, `scikit-image`: Processamento de imagens
- `jupyter`: Notebooks interativos

### Passo 3: Verificar Datasets

- **Dados Tabulares**: Certifique-se de que o arquivo `data/tabular/breast-cancer.csv` está presente
- **Dados de Imagens**: Os datasets serão baixados automaticamente ao executar os notebooks de exploração (03 e 05)

### Passo 4: Iniciar Jupyter

```bash
jupyter notebook
```

Ou, se preferir JupyterLab:

```bash
jupyter lab
```

### Passo 5: Instalar Dependências de Desenvolvimento (Opcional)

Para executar os testes do projeto:

```bash
pip3 install -r requirements-dev.txt
```

**Nota**: Se você receber um erro "command not found: pip", use `pip3` em vez de `pip`.

---

## 🧪 Executando Testes

O projeto inclui uma suíte completa de testes seguindo as melhores práticas de mercado.

### Executar Todos os Testes

```bash
pytest
```

### Executar Testes com Cobertura

```bash
pytest --cov=src --cov-report=html
```

Isso gerará um relatório HTML em `htmlcov/index.html` mostrando a cobertura de código.

### Executar Apenas Testes Unitários

```bash
pytest tests/unit -m unit
```

### Executar Apenas Testes de Integração

```bash
pytest tests/integration -m integration
```

### Executar Testes Específicos

```bash
# Testar um módulo específico
pytest tests/unit/test_tabular_processing.py

# Testar uma classe específica
pytest tests/unit/test_tabular_processing.py::TestSplitData

# Testar uma função específica
pytest tests/unit/test_tabular_processing.py::TestSplitData::test_split_data_basic
```

### Ver Cobertura de Código

```bash
# Cobertura no terminal
pytest --cov=src --cov-report=term-missing

# Cobertura em HTML (abre no navegador)
pytest --cov=src --cov-report=html && open htmlcov/index.html
```

### Estrutura de Testes

```
tests/
├── conftest.py              # Fixtures compartilhadas
├── unit/                     # Testes unitários
│   ├── test_tabular_processing.py
│   ├── test_tabular_evaluate.py
│   ├── test_vision_data_loader.py
│   ├── test_vision_preprocessing.py
│   ├── test_vision_models.py
│   └── test_vision_evaluation.py
├── integration/             # Testes de integração
│   ├── test_tabular_pipeline.py
│   └── test_vision_pipeline.py
└── fixtures/                # Dados sintéticos para testes
    ├── sample_data.py
    └── sample_images.py
```

### Cobertura de Testes

Os testes cobrem:
- ✅ Todas as funções dos módulos `src/tabular/`
- ✅ Todas as funções dos módulos `src/vision/`
- ✅ Edge cases e tratamento de erros
- ✅ Validação de dados de entrada
- ✅ Testes de integração para pipelines completos
- ✅ Mocks para operações custosas (downloads, treinamento)

**Meta de cobertura**: 80% ou mais

---

## 📚 Guia Passo a Passo Completo

Este guia explica detalhadamente cada notebook do projeto, o que ele faz, o que você verá ao executá-lo e como interpretar os resultados.

### Notebook 01: Exploração de Dados Tabulares

#### 🎯 Objetivo

Este notebook realiza uma **análise exploratória de dados (EDA)** do dataset de câncer de mama. Ele examina as características dos dados, identifica padrões, verifica a qualidade dos dados e prepara o terreno para a modelagem.

#### 📋 Pré-requisitos

- Python 3.8+ instalado
- Dependências do `requirements.txt` instaladas
- Arquivo `data/tabular/breast-cancer.csv` presente no projeto

#### 📝 Passo a Passo

**Passo 1: Carregamento dos Dados**
- **O que fazer**: Execute a primeira célula que importa as bibliotecas e carrega o dataset
- **O que você verá**: Uma tabela mostrando as primeiras 5 linhas do dataset com todas as colunas
- **O que significa**: Você está visualizando uma amostra dos dados. Cada linha representa um paciente e cada coluna uma característica medida (raio, textura, perímetro, etc.)

**Passo 2: Análise Descritiva**
- **O que fazer**: Execute as células que mostram `df.info()` e `df.describe()`
- **O que você verá**: 
  - `df.info()`: Lista de todas as colunas, tipos de dados e quantidade de valores não nulos
  - `df.describe()`: Estatísticas descritivas (média, desvio padrão, mínimo, máximo, quartis) para cada coluna numérica
- **O que significa**: 
  - `info()` confirma que não há valores faltantes (todos os 569 registros têm valores)
  - `describe()` mostra a distribuição dos valores. Por exemplo, se a média de `radius_mean` é 14.1, isso indica o tamanho médio dos núcleos celulares

**Passo 3: Análise da Variável Alvo**
- **O que fazer**: Execute as células que visualizam a distribuição da variável `diagnosis`
- **O que você verá**: 
  - Um gráfico de barras mostrando quantos casos são Benignos (B) e quantos são Malignos (M)
  - Um gráfico de pizza (pie chart) mostrando as proporções
  - Estatísticas de contagem
- **O que significa**: 
  - Você verá aproximadamente 357 casos Benignos (62.7%) e 212 casos Malignos (37.3%)
  - Isso indica um **desbalanceamento moderado** das classes, o que é importante considerar na modelagem

**Passo 4: Análise de Correlação**
- **O que fazer**: Execute as células que criam a matriz de correlação
- **O que você verá**: 
  - Um mapa de calor (heatmap) colorido mostrando correlações entre variáveis
  - Cores quentes (vermelho/laranja) indicam correlação positiva forte
  - Cores frias (azul) indicam correlação negativa
- **O que significa**: 
  - Variáveis altamente correlacionadas (ex: `radius_mean` e `perimeter_mean`) fornecem informações similares
  - Isso pode indicar redundância, mas também pode ser útil para o modelo

**Passo 5: Visualização de Distribuições**
- **O que fazer**: Execute as células que criam histogramas e boxplots
- **O que você verá**: 
  - Histogramas mostrando a distribuição de valores para diferentes features
  - Boxplots comparando a distribuição entre classes Benignas e Malignas
- **O que significa**: 
  - Se você vê diferenças claras nos boxplots entre B e M, essa feature é provavelmente importante para classificação
  - Por exemplo, se `radius_worst` é maior em casos Malignos, isso faz sentido clinicamente

#### 📊 Saídas Esperadas

1. **Tabela de dados**: Primeiras linhas do dataset
2. **Estatísticas descritivas**: Tabela com médias, desvios padrão, etc.
3. **Gráfico de distribuição de classes**: Bar chart e pie chart mostrando ~62% Benigno, ~38% Maligno
4. **Matriz de correlação**: Heatmap colorido mostrando relações entre variáveis
5. **Histogramas**: Distribuições de features individuais
6. **Boxplots**: Comparação de features entre classes B e M

#### 💡 Interpretação dos Resultados

- **Qualidade dos dados**: Se não há valores faltantes e os tipos de dados estão corretos, os dados estão prontos para modelagem
- **Desbalanceamento**: O dataset tem mais casos benignos que malignos. Isso é normal, mas devemos usar estratificação na divisão dos dados
- **Features importantes**: Features que mostram diferenças claras entre B e M nos boxplots são candidatas a serem importantes para o modelo
- **Correlações**: Features muito correlacionadas (ex: radius, perimeter, area) são relacionadas, o que é esperado

#### ➡️ Próximos Passos

Após completar este notebook, você está pronto para o **Notebook 02: Modelagem de Dados Tabulares**, onde os dados serão usados para treinar modelos de machine learning.

---

### Notebook 02: Modelagem de Dados Tabulares

#### 🎯 Objetivo

Este notebook treina e avalia modelos de machine learning para prever se um tumor é benigno ou maligno com base nas características clínicas. Você verá dois modelos diferentes (Regressão Logística e Random Forest) sendo treinados, comparados e interpretados.

#### 📋 Pré-requisitos

- Notebook 01 executado (para entender os dados)
- Dataset carregado e limpo
- Bibliotecas scikit-learn e SHAP instaladas

#### 📝 Passo a Passo

**Passo 1: Preparação dos Dados**
- **O que fazer**: Execute as células que separam features (X) da variável alvo (y) e dividem os dados
- **O que você verá**: 
  - Mensagens mostrando quantas features foram selecionadas (30)
  - Informações sobre a divisão: 341 amostras de treino, 114 de validação, 114 de teste
- **O que significa**: 
  - Os dados são divididos em três conjuntos: **treino** (para aprender), **validação** (para ajustar) e **teste** (para avaliar final)
  - A divisão é **estratificada**, mantendo a proporção de classes em cada conjunto

**Passo 2: Treinamento do Modelo 1 - Regressão Logística**
- **O que fazer**: Execute as células que criam e treinam o modelo de Regressão Logística
- **O que você verá**: 
  - Mensagem "Treinando Regressão Logística..."
  - Relatórios de classificação mostrando métricas para validação e teste
  - Tabelas com Precision, Recall, F1-Score para cada classe
- **O que significa**: 
  - **Precision**: Quando o modelo diz "Maligno", quantas vezes está correto (~97%)
  - **Recall**: Quantos casos malignos o modelo consegue detectar (~93%)
  - **F1-Score**: Média balanceada entre Precision e Recall (~95%)

**Passo 3: Treinamento do Modelo 2 - Random Forest**
- **O que fazer**: Execute as células que criam e treinam o modelo Random Forest
- **O que você verá**: 
  - Mensagem "Treinando Random Forest..."
  - Relatórios de classificação similares ao modelo anterior
  - Métricas geralmente ligeiramente melhores
- **O que significa**: 
  - Random Forest é um modelo mais complexo que combina múltiplas árvores de decisão
  - Geralmente apresenta melhor desempenho, mas é menos interpretável

**Passo 4: Comparação dos Modelos**
- **O que fazer**: Execute as células que comparam os dois modelos
- **O que você verá**: 
  - Uma tabela comparativa mostrando Accuracy, Precision, Recall e F1-Score lado a lado
  - Um gráfico de barras comparando as métricas
  - Identificação do melhor modelo
- **O que significa**: 
  - Random Forest geralmente apresenta Accuracy ~97.4% vs ~96.5% da Regressão Logística
  - O melhor modelo é selecionado para uso futuro

**Passo 5: Matriz de Confusão**
- **O que fazer**: Execute as células que geram a matriz de confusão
- **O que você verá**: 
  - Uma matriz 2x2 mostrando:
    - **Verdadeiros Negativos (TN)**: Casos benignos corretamente identificados
    - **Falsos Positivos (FP)**: Casos benignos classificados como malignos (alarmes falsos)
    - **Falsos Negativos (FN)**: Casos malignos classificados como benignos (perigosos!)
    - **Verdadeiros Positivos (TP)**: Casos malignos corretamente identificados
- **O que significa**: 
  - **Falsos Negativos são críticos**: Um caso maligno não detectado pode ser perigoso
  - O modelo ideal tem poucos ou nenhum falso negativo

**Passo 6: Feature Importance**
- **O que fazer**: Execute as células que mostram a importância das features
- **O que você verá**: 
  - Um gráfico de barras horizontal mostrando as features mais importantes
  - Top 10-15 features listadas com suas importâncias
- **O que significa**: 
  - Features como `concave points_worst` e `perimeter_worst` são as mais importantes
  - Isso indica que características de concavidade e tamanho são mais preditivas

**Passo 7: Análise SHAP**
- **O que fazer**: Execute as células que calculam e visualizam valores SHAP
- **O que você verá**: 
  - **Summary Plot**: Um gráfico mostrando como cada feature afeta as predições
  - **Bar Plot**: Importância média das features segundo SHAP
  - **Waterfall Plot**: Explicação de uma predição específica
- **O que significa**: 
  - SHAP explica **por que** o modelo fez cada predição
  - Valores SHAP positivos (vermelho) aumentam a probabilidade de "Maligno"
  - Valores SHAP negativos (azul) diminuem a probabilidade de "Maligno"

**Passo 8: Discussão Crítica**
- **O que fazer**: Leia as células de discussão sobre limitações e considerações éticas
- **O que você verá**: 
  - Lista de limitações do modelo
  - Considerações sobre uso prático
  - Considerações éticas e médicas
- **O que significa**: 
  - O modelo tem limitações (dataset pequeno, não considera contexto completo)
  - **Nunca deve substituir o diagnóstico médico**
  - Deve ser usado apenas como ferramenta de apoio

#### 📊 Saídas Esperadas

1. **Relatórios de classificação**: Tabelas com métricas para cada modelo
2. **Gráfico comparativo**: Barras mostrando Accuracy, Precision, Recall, F1-Score
3. **Matriz de confusão**: Visualização 2x2 dos acertos e erros
4. **Feature importance**: Gráfico de barras com top features
5. **SHAP Summary Plot**: Visualização da importância global das features
6. **SHAP Bar Plot**: Importância média segundo SHAP
7. **SHAP Waterfall Plot**: Explicação de uma predição específica

#### 💡 Interpretação dos Resultados

- **Métricas altas (>95%)**: Indicam que o modelo está funcionando bem, mas lembre-se: em medicina, até 1% de erro pode ser significativo
- **Recall de ~93%**: Significa que 7% dos casos malignos não são detectados. Isso é crítico e precisa ser melhorado
- **Precision de 100% (Random Forest)**: Significa que quando o modelo diz "maligno", está sempre correto - não há falsos alarmes
- **Feature importance**: Confirma que características de tamanho e forma são mais importantes que textura
- **SHAP**: Fornece transparência sobre as decisões do modelo, essencial para confiança médica

#### ➡️ Próximos Passos

Após completar este notebook, você pode:
- Explorar os notebooks de visão computacional (03-06) para classificação de imagens
- Usar o modelo treinado para fazer predições em novos dados
- Ajustar hiperparâmetros para melhorar o desempenho

---

### Notebook 03: Exploração de Imagens de Pneumonia

#### 🎯 Objetivo

Este notebook realiza uma análise exploratória do dataset de imagens de raio-X de tórax para detecção de pneumonia. Ele baixa o dataset, explora sua estrutura, visualiza amostras de imagens e analisa a distribuição das classes.

#### 📋 Pré-requisitos

- Python 3.8+ instalado
- Dependências instaladas (especialmente `kagglehub` para download)
- Conexão com internet (para baixar o dataset do Kaggle)

#### 📝 Passo a Passo

**Passo 1: Download do Dataset**
- **O que fazer**: Execute a célula que baixa o dataset do Kaggle
- **O que você verá**: 
  - Mensagens de progresso do download
  - Caminho onde o dataset foi salvo
  - Pode levar alguns minutos dependendo da conexão
- **O que significa**: 
  - O dataset será baixado automaticamente usando `kagglehub`
  - As imagens serão organizadas em pastas: `train/NORMAL/`, `train/PNEUMONIA/`, `test/`, `val/`

**Passo 2: Análise da Estrutura**
- **O que fazer**: Execute as células que analisam a estrutura de diretórios
- **O que você verá**: 
  - Contagem de imagens em cada pasta
  - Distribuição entre classes (Normal vs Pneumonia)
  - Estrutura de diretórios
- **O que significa**: 
  - Você verá milhares de imagens (ex: ~1300 Normal, ~3900 Pneumonia no treino)
  - Há um desbalanceamento significativo (mais casos de pneumonia)
  - Os dados já vêm divididos em treino/teste/validação

**Passo 3: Visualização de Amostras**
- **O que fazer**: Execute as células que mostram imagens de exemplo
- **O que você verá**: 
  - Grid de imagens mostrando exemplos de cada classe
  - Imagens de raio-X de tórax em escala de cinza
  - Labels indicando "Normal" ou "Pneumonia"
- **O que significa**: 
  - **Normal**: Pulmões limpos, sem opacidades
  - **Pneumonia**: Opacidades brancas (infiltrados) indicando infecção
  - As diferenças podem ser sutis, o que torna o problema desafiador

**Passo 4: Análise de Dimensões**
- **O que fazer**: Execute as células que verificam as dimensões das imagens
- **O que você verá**: 
  - Estatísticas sobre largura, altura e formato das imagens
  - Algumas imagens podem ter tamanhos diferentes
- **O que significa**: 
  - As imagens precisarão ser redimensionadas para um tamanho uniforme antes do treinamento
  - Geralmente redimensionamos para 224x224 pixels

**Passo 5: Análise de Qualidade**
- **O que fazer**: Execute as células que verificam a qualidade das imagens
- **O que você verá**: 
  - Verificação de imagens corrompidas ou inválidas
  - Estatísticas sobre canais de cor (RGB vs escala de cinza)
- **O que significa**: 
  - A maioria das imagens de raio-X são em escala de cinza, mas algumas podem ter 3 canais
  - Imagens corrompidas serão identificadas e podem ser removidas

**Passo 6: Distribuição de Classes**
- **O que fazer**: Execute as células que visualizam a distribuição
- **O que você verá**: 
  - Gráficos de barras mostrando contagem por classe
  - Gráficos de pizza mostrando proporções
- **O que significa**: 
  - Há mais imagens de pneumonia que normais (desbalanceamento)
  - Isso será tratado durante o treinamento com técnicas como data augmentation e class weights

#### 📊 Saídas Esperadas

1. **Mensagens de download**: Progresso do download do dataset
2. **Estatísticas de estrutura**: Contagem de imagens por pasta e classe
3. **Grid de imagens**: Visualização de amostras de cada classe
4. **Análise de dimensões**: Estatísticas sobre tamanhos das imagens
5. **Gráficos de distribuição**: Barras e pizza mostrando proporções de classes

#### 💡 Interpretação dos Resultados

- **Dataset grande**: Milhares de imagens fornecem dados suficientes para treinar uma CNN
- **Desbalanceamento**: Mais casos de pneumonia é esperado em um dataset médico real
- **Qualidade variável**: Imagens podem ter diferentes resoluções e qualidades, o que é normal
- **Diferenças sutis**: As diferenças entre Normal e Pneumonia podem ser difíceis de ver a olho nu, mas o modelo aprenderá padrões

#### ➡️ Próximos Passos

Após completar este notebook, você está pronto para o **Notebook 04: Modelagem CNN para Pneumonia**, onde uma rede neural convolucional será treinada para classificar as imagens.

---

### Notebook 04: Modelagem CNN para Pneumonia

#### 🎯 Objetivo

Este notebook treina uma **Rede Neural Convolucional (CNN)** para classificar imagens de raio-X de tórax como Normal ou Pneumonia. Você verá o processo completo: pré-processamento, treinamento, avaliação e interpretabilidade com Grad-CAM.

#### 📋 Pré-requisitos

- Notebook 03 executado (dataset baixado e explorado)
- TensorFlow/Keras instalado
- GPU opcional (mas recomendado para treinamento mais rápido)

#### 📝 Passo a Passo

**Passo 1: Carregamento e Divisão dos Dados**
- **O que fazer**: Execute as células que carregam as imagens e dividem em treino/validação/teste
- **O que você verá**: 
  - Mensagens mostrando quantas imagens foram carregadas
  - Informações sobre a divisão: 60% treino, 20% validação, 20% teste
  - Estatísticas de distribuição de classes
- **O que significa**: 
  - As imagens são carregadas e organizadas em batches para eficiência
  - A divisão mantém a proporção de classes (estratificação)

**Passo 2: Data Augmentation**
- **O que fazer**: Execute as células que configuram data augmentation
- **O que você verá**: 
  - Configurações de transformações: rotação, zoom, deslocamento, etc.
- **O que significa**: 
  - **Data augmentation** cria variações das imagens (rotacionadas, ampliadas, etc.)
  - Isso aumenta a diversidade do dataset e reduz overfitting
  - Apenas aplicado no conjunto de treino

**Passo 3: Criação do Modelo CNN**
- **O que fazer**: Execute as células que criam a arquitetura da CNN
- **O que você verá**: 
  - Resumo da arquitetura mostrando todas as camadas
  - Número total de parâmetros (milhões)
  - Estrutura: camadas convolucionais → pooling → camadas densas
- **O que significa**: 
  - **Camadas convolucionais**: Detectam padrões (bordas, texturas, formas)
  - **Pooling**: Reduz dimensão, mantendo informações importantes
  - **Camadas densas**: Fazem a classificação final

**Passo 4: Compilação do Modelo**
- **O que fazer**: Execute as células que compilam o modelo
- **O que você verá**: 
  - Configurações: otimizador (Adam), função de loss, métricas
- **O que significa**: 
  - **Adam**: Algoritmo de otimização eficiente
  - **Categorical Crossentropy**: Função de loss adequada para classificação
  - **Métricas**: Accuracy, Precision, Recall serão monitoradas

**Passo 5: Treinamento**
- **O que fazer**: Execute a célula que inicia o treinamento
- **O que você verá**: 
  - Progresso por época mostrando:
    - Loss (erro) no treino e validação
    - Accuracy no treino e validação
    - Tempo por época
  - Pode levar de 30 minutos a várias horas dependendo do hardware
- **O que significa**: 
  - O modelo está aprendendo a distinguir Normal de Pneumonia
  - **Loss diminuindo**: O modelo está melhorando
  - **Accuracy aumentando**: O modelo está acertando mais
  - **Early stopping**: O treinamento para automaticamente se não melhorar

**Passo 6: Visualização do Histórico de Treinamento**
- **O que fazer**: Execute as células que plotam gráficos do histórico
- **O que você verá**: 
  - Gráficos de Loss (treino vs validação) ao longo das épocas
  - Gráficos de Accuracy (treino vs validação) ao longo das épocas
- **O que significa**: 
  - **Curvas convergindo**: O modelo está aprendendo bem
  - **Gap grande entre treino e validação**: Possível overfitting
  - **Validação melhorando**: O modelo está generalizando bem

**Passo 7: Avaliação no Conjunto de Teste**
- **O que fazer**: Execute as células que avaliam o modelo no conjunto de teste
- **O que você verá**: 
  - Métricas finais: Accuracy, Precision, Recall, F1-Score
  - Matriz de confusão
  - Curva ROC e AUC
- **O que significa**: 
  - **Accuracy > 80%**: Bom desempenho para uma CNN simples
  - **Matriz de confusão**: Mostra quantos casos foram classificados corretamente
  - **ROC-AUC**: Mede a capacidade de distinguir entre classes (quanto maior, melhor)

**Passo 8: Visualização de Predições**
- **O que fazer**: Execute as células que mostram predições em imagens de teste
- **O que você verá**: 
  - Grid de imagens com predições
  - Labels mostrando: Classe verdadeira vs Predição vs Confiança
  - Imagens corretas e incorretas destacadas
- **O que significa**: 
  - **Confiança alta (>90%)**: O modelo está muito certo
  - **Confiança baixa (<70%)**: O modelo está incerto
  - **Erros**: Casos difíceis que o modelo confundiu

**Passo 9: Grad-CAM (Interpretabilidade)**
- **O que fazer**: Execute as células que geram visualizações Grad-CAM
- **O que você verá**: 
  - Imagens originais lado a lado com heatmaps coloridos
  - Regiões em vermelho/laranja: áreas que o modelo considera importantes
  - Superposição do heatmap na imagem original
- **O que significa**: 
  - **Grad-CAM** mostra **onde** o modelo está olhando
  - Regiões destacadas devem corresponder a áreas clinicamente relevantes (pulmões)
  - Se o modelo foca em áreas irrelevantes, pode indicar problemas

#### 📊 Saídas Esperadas

1. **Resumo da arquitetura**: Estrutura completa da CNN
2. **Progresso de treinamento**: Métricas por época
3. **Gráficos de histórico**: Loss e Accuracy ao longo do tempo
4. **Métricas finais**: Tabela com Accuracy, Precision, Recall, F1-Score
5. **Matriz de confusão**: Visualização 2x2 dos acertos e erros
6. **Curva ROC**: Gráfico mostrando performance de classificação
7. **Grid de predições**: Imagens com predições e confianças
8. **Grad-CAM heatmaps**: Visualizações mostrando regiões importantes

#### 💡 Interpretação dos Resultados

- **Accuracy > 80%**: Bom desempenho, mas em medicina sempre buscamos melhorar
- **Recall alto**: Importante para não perder casos de pneumonia
- **Grad-CAM focado nos pulmões**: Indica que o modelo está aprendendo padrões corretos
- **Overfitting**: Se accuracy de treino >> accuracy de validação, o modelo está decorando os dados
- **Tempo de treinamento**: CNNs são computacionalmente intensivas, mas os resultados valem a pena

#### ➡️ Próximos Passos

Após completar este notebook, você pode:
- Explorar os notebooks de câncer de mama (05-06)
- Experimentar diferentes arquiteturas de CNN
- Ajustar hiperparâmetros para melhorar o desempenho

---

### Notebook 05: Exploração de Mamografias

#### 🎯 Objetivo

Este notebook realiza uma análise exploratória do dataset de mamografias (CBIS-DDSM) para detecção de câncer de mama. Similar ao notebook 03, mas focado em imagens de mamografia.

#### 📋 Pré-requisitos

- Python 3.8+ instalado
- Dependências instaladas (especialmente `kagglehub`)
- Conexão com internet
- **Nota**: Este dataset é maior e pode levar mais tempo para baixar

#### 📝 Passo a Passo

**Passo 1: Download do Dataset**
- **O que fazer**: Execute a célula que baixa o dataset CBIS-DDSM
- **O que você verá**: 
  - Mensagens de progresso (pode levar 10-30 minutos)
  - Caminho onde o dataset foi salvo
  - Estrutura de diretórios complexa (o dataset CBIS-DDSM tem estrutura aninhada)
- **O que significa**: 
  - Este dataset é maior e mais complexo que o de pneumonia
  - As imagens são de alta resolução (mamografias detalhadas)
  - Estrutura: `train/BENIGN/`, `train/MALIGNANT/`, etc.

**Passo 2: Análise da Estrutura**
- **O que fazer**: Execute as células que analisam a estrutura
- **O que você verá**: 
  - Contagem de imagens por classe
  - Estrutura de diretórios (pode ser aninhada)
  - Estatísticas de distribuição
- **O que significa**: 
  - Dataset pode ter centenas ou milhares de imagens
  - Distribuição entre Benigno e Maligno
  - Estrutura pode requerer navegação em subdiretórios

**Passo 3: Visualização de Amostras**
- **O que fazer**: Execute as células que mostram imagens de exemplo
- **O que você verá**: 
  - Grid de mamografias em escala de cinza
  - Imagens de alta resolução mostrando tecido mamário
  - Labels indicando "Benigno" ou "Maligno"
- **O que significa**: 
  - **Mamografias**: Imagens de raio-X das mamas
  - **Lesões benignas**: Massas não cancerosas
  - **Lesões malignas**: Câncer de mama
  - Diferenças podem ser muito sutis e requerem análise especializada

**Passo 4: Análise de Dimensões e Qualidade**
- **O que fazer**: Execute as células que verificam dimensões e qualidade
- **O que você verá**: 
  - Estatísticas sobre tamanhos das imagens (geralmente grandes, ex: 2000x3000 pixels)
  - Verificação de imagens corrompidas
  - Informações sobre formato (geralmente DICOM ou PNG)
- **O que significa**: 
  - Imagens de alta resolução precisarão ser redimensionadas para treinamento (ex: 256x256)
  - Formato DICOM é comum em imagens médicas e pode requerer conversão

**Passo 5: Distribuição de Classes**
- **O que fazer**: Execute as células que visualizam a distribuição
- **O que você verá**: 
  - Gráficos mostrando proporção de Benigno vs Maligno
  - Estatísticas de contagem
- **O que significa**: 
  - Pode haver desbalanceamento (mais casos benignos é comum)
  - Isso será tratado durante o treinamento

#### 📊 Saídas Esperadas

1. **Mensagens de download**: Progresso (pode ser longo)
2. **Estatísticas de estrutura**: Contagem e organização de imagens
3. **Grid de mamografias**: Visualização de amostras
4. **Análise de dimensões**: Tamanhos das imagens (geralmente grandes)
5. **Gráficos de distribuição**: Proporções de classes

#### 💡 Interpretação dos Resultados

- **Dataset complexo**: Estrutura aninhada é comum em datasets médicos profissionais
- **Alta resolução**: Imagens detalhadas são importantes para detectar lesões pequenas
- **Diferenças sutis**: Distinguir benigno de maligno é desafiador mesmo para especialistas
- **Desbalanceamento**: Mais casos benignos é esperado em dados reais

#### ➡️ Próximos Passos

Após completar este notebook, você está pronto para o **Notebook 06: Modelagem CNN para Câncer de Mama**, onde uma CNN será treinada para classificar as mamografias.

---

### Notebook 06: Modelagem CNN para Câncer de Mama

#### 🎯 Objetivo

Este notebook treina uma **CNN** para classificar mamografias como Benignas ou Malignas. Similar ao notebook 04, mas adaptado para imagens em escala de cinza e com arquitetura otimizada para o problema específico.

#### 📋 Pré-requisitos

- Notebook 05 executado (dataset baixado)
- TensorFlow/Keras instalado
- GPU recomendado (treinamento pode ser longo)

#### 📝 Passo a Passo

**Passo 1: Carregamento e Pré-processamento**
- **O que fazer**: Execute as células que carregam e preprocessam as imagens
- **O que você verá**: 
  - Mensagens sobre carregamento de imagens
  - Conversão para escala de cinza (1 canal em vez de 3)
  - Redimensionamento para 256x256 pixels
  - Divisão em treino/validação/teste
- **O que significa**: 
  - Mamografias são naturalmente em escala de cinza
  - Redimensionamento é necessário para eficiência computacional
  - Tamanho 256x256 é um bom equilíbrio entre detalhe e velocidade

**Passo 2: Data Augmentation**
- **O que fazer**: Execute as células que configuram augmentation
- **O que você verá**: 
  - Configurações similares ao notebook 04, mas adaptadas para escala de cinza
  - Rotação, zoom, deslocamento, brightness adjustment
- **O que significa**: 
  - Augmentation é especialmente importante para datasets menores
  - Variações de brilho simulam diferentes condições de imagem

**Passo 3: Criação do Modelo CNN**
- **O que fazer**: Execute as células que criam a arquitetura
- **O que você verá**: 
  - Arquitetura com 5 blocos convolucionais (mais profunda que pneumonia)
  - Global Average Pooling (técnica avançada para reduzir overfitting)
  - Batch Normalization e Dropout para regularização
- **O que significa**: 
  - Arquitetura mais profunda captura padrões mais complexos
  - Técnicas de regularização previnem overfitting
  - Global Average Pooling reduz parâmetros e melhora generalização

**Passo 4: Compilação com Focal Loss (Opcional)**
- **O que fazer**: Execute as células que compilam o modelo
- **O que você verá**: 
  - Opção de usar Focal Loss ou Categorical Crossentropy
  - Focal Loss é especialmente útil para classes desbalanceadas
- **O que significa**: 
  - **Focal Loss**: Foca em exemplos difíceis, útil quando há desbalanceamento
  - **Class Weights**: Ajusta a importância de cada classe durante treinamento

**Passo 5: Treinamento**
- **O que fazer**: Execute a célula que inicia o treinamento
- **O que você verá**: 
  - Progresso similar ao notebook 04
  - Pode levar mais tempo devido à arquitetura mais profunda
  - Early stopping e redução de learning rate automáticos
- **O que significa**: 
  - O modelo está aprendendo padrões sutis em mamografias
  - Callbacks automáticos otimizam o treinamento

**Passo 6: Avaliação e Métricas**
- **O que fazer**: Execute as células de avaliação
- **O que você verá**: 
  - Métricas completas: Accuracy, Precision, Recall, F1-Score
  - Matriz de confusão
  - Curva ROC
  - Análise por classe
- **O que significa**: 
  - **Recall alto para Maligno**: Crítico para não perder casos de câncer
  - **Precision alta**: Evita alarmes falsos e biópsias desnecessárias
  - Métricas balanceadas indicam bom desempenho geral

**Passo 7: Grad-CAM**
- **O que fazer**: Execute as células que geram Grad-CAM
- **O que você verá**: 
  - Heatmaps mostrando regiões importantes nas mamografias
  - Superposição nas imagens originais
  - Análise de casos corretos e incorretos
- **O que significa**: 
  - Regiões destacadas devem corresponder a lesões suspeitas
  - Se o modelo foca em áreas irrelevantes, pode indicar problemas
  - Grad-CAM é essencial para validação clínica

**Passo 8: Validação e Discussão**
- **O que fazer**: Leia as células de discussão sobre resultados
- **O que você verá**: 
  - Análise crítica do desempenho
  - Limitações do modelo
  - Considerações para uso clínico
- **O que significa**: 
  - Mesmo com alta accuracy, o modelo tem limitações
  - **Nunca deve substituir diagnóstico médico**
  - Pode ser usado como ferramenta de triagem/apoio

#### 📊 Saídas Esperadas

1. **Resumo da arquitetura**: CNN com 5 blocos convolucionais
2. **Progresso de treinamento**: Métricas por época
3. **Gráficos de histórico**: Loss e Accuracy
4. **Métricas finais**: Tabela completa de avaliação
5. **Matriz de confusão**: Performance detalhada
6. **Curva ROC**: Capacidade de classificação
7. **Grad-CAM heatmaps**: Regiões importantes nas mamografias

#### 💡 Interpretação dos Resultados

- **Accuracy > 80%**: Bom, mas em câncer sempre buscamos melhorar
- **Recall para Maligno > 90%**: Essencial - não podemos perder casos de câncer
- **Grad-CAM focado em lesões**: Valida que o modelo está aprendendo padrões corretos
- **Focal Loss**: Pode melhorar performance em classes desbalanceadas
- **Arquitetura profunda**: Captura padrões complexos, mas requer mais dados

#### ➡️ Próximos Passos

Após completar todos os notebooks, você pode:
- Comparar resultados entre diferentes abordagens
- Experimentar transfer learning (usar modelos pré-treinados)
- Ajustar hiperparâmetros para otimização
- Integrar os modelos em uma aplicação

---

## 🔬 Detalhes Técnicos

Esta seção apresenta os detalhes técnicos do projeto, incluindo estratégias de pré-processamento, justificativas dos modelos e discussões sobre métricas.

### Estratégias de Pré-processamento

#### Dados Tabulares

**1. Limpeza de Dados**

- **Remoção de colunas não relevantes**:
  - `id`: Identificador único (não preditivo)
  - `Unnamed: 32`: Coluna vazia/duplicada

- **Tratamento de valores ausentes e infinitos**:
  - Substituição de valores infinitos por NaN
  - Preenchimento de NaN com a média da coluna (se necessário)
  - No dataset utilizado, não foram encontrados valores ausentes

- **Seleção de features**:
  - Utilização apenas de colunas numéricas
  - Remoção de colunas identificadoras

**2. Normalização**

- **StandardScaler**: Normalização das features para média zero e desvio padrão unitário
- **Justificativa**:
  - Diferentes features têm escalas distintas (ex: área vs. textura)
  - Modelos lineares (Regressão Logística) são sensíveis à escala
  - Facilita convergência e melhora desempenho

**3. Divisão dos Dados**

- **Estratégia**: Divisão estratificada em três conjuntos
  - **Treino (60%)**: 341 amostras - Para treinar os modelos
  - **Validação (20%)**: 114 amostras - Para ajuste de hiperparâmetros e seleção de modelo
  - **Teste (20%)**: 114 amostras - Para avaliação final e relatório de desempenho
- **Estratificação**: Mantém a proporção de classes em cada conjunto
- **Random State**: 42 (para reprodutibilidade)

#### Dados de Imagens

**1. Redimensionamento e Normalização**

- **Redimensionamento**: Todas as imagens foram redimensionadas para tamanhos fixos
  - Pneumonia: 224x224 pixels (RGB)
  - Câncer de Mama: 256x256 pixels (escala de cinza)
- **Normalização**: Pixels normalizados para o intervalo [0, 1] dividindo por 255
- **Conversão de Cores**:
  - Pneumonia: Mantido RGB (3 canais)
  - Câncer de Mama: Convertido para escala de cinza (1 canal)

**2. Data Augmentation**

Para aumentar a robustez do modelo e reduzir overfitting, foram aplicadas técnicas de data augmentation no conjunto de treino:

- **Rotação**: ±30 graus
- **Deslocamento**: Horizontal e vertical (±15%)
- **Zoom**: ±20%
- **Flip Horizontal**: Espelhamento aleatório
- **Flip Vertical**: Espelhamento vertical (para câncer de mama)
- **Brightness**: Ajuste de brilho [0.8, 1.2]
- **Shear**: Cisalhamento de ±10%

**Justificativa**:
- Aumenta a diversidade do conjunto de treino
- Melhora generalização
- Simula variações naturais em imagens médicas (posicionamento, ângulo, etc.)

**3. Divisão dos Dados**

- **Treino (60%)**: Para treinar o modelo
- **Validação (20%)**: Para ajuste de hiperparâmetros e early stopping
- **Teste (20%)**: Para avaliação final
- **Estratificação**: Mantém proporção de classes em cada conjunto

### Modelos Utilizados e Justificativa

#### Dados Tabulares

**1. Regressão Logística**

**Justificativa**:
- Modelo linear interpretável e eficiente
- Funciona bem como baseline para comparação
- Rápido para treinar e fazer predições
- Boa performance em problemas de classificação binária
- Probabilidades de saída são calibradas

**Parâmetros**:
- `solver='lbfgs'`: Algoritmo robusto para problemas pequenos/médios
- `C=1.0`: Regularização L2 (inverso da força de regularização)
- `max_iter=1000`: Número máximo de iterações
- `random_state=42`: Reprodutibilidade

**Vantagens**:
- Interpretabilidade (coeficientes lineares)
- Baixa complexidade computacional
- Menor risco de overfitting

**Desvantagens**:
- Assume relação linear entre features e target
- Pode não capturar interações complexas

**2. Random Forest**

**Justificativa**:
- Algoritmo de ensemble robusto e poderoso
- Capaz de capturar relações não-lineares
- Menos propenso a overfitting que árvores individuais
- Fornece feature importance nativa
- Geralmente apresenta melhor desempenho que modelos lineares

**Parâmetros**:
- `n_estimators=100`: Número de árvores no ensemble
- `max_depth=10`: Profundidade máxima das árvores (controla complexidade)
- `random_state=42`: Reprodutibilidade

**Vantagens**:
- Alta capacidade de modelagem
- Robustez a outliers
- Feature importance integrada
- Boa performance geral

**Desvantagens**:
- Menos interpretável que modelos lineares
- Maior complexidade computacional
- Pode ser mais difícil de explicar para não-especialistas

**3. Pipeline de Processamento**

Ambos os modelos foram implementados em um pipeline que inclui:

1. **StandardScaler**: Normalização das features
2. **Modelo**: Regressão Logística ou Random Forest

Isso garante que:
- Novos dados sejam pré-processados da mesma forma
- O modelo salvo inclui todas as transformações necessárias

#### Dados de Imagens (CNNs)

**1. CNN para Pneumonia**

**Arquitetura**:
- **Input**: Imagens RGB 224x224x3
- **4 Blocos Convolucionais**:
  - Bloco 1: 32 filtros 3x3 + BatchNorm + MaxPooling 2x2 + Dropout 0.25
  - Bloco 2: 64 filtros 3x3 + BatchNorm + MaxPooling 2x2 + Dropout 0.25
  - Bloco 3: 128 filtros 3x3 + BatchNorm + MaxPooling 2x2 + Dropout 0.25
  - Bloco 4: 128 filtros 3x3 + BatchNorm + MaxPooling 2x2 + Dropout 0.25
- **Camadas Densas**:
  - Flatten
  - Dense(512) + BatchNorm + Dropout(0.5)
  - Dense(256) + BatchNorm + Dropout(0.5)
  - Dense(2, activation='softmax')

**Total de parâmetros**: ~2-3 milhões

**2. CNN para Câncer de Mama**

**Arquitetura**:
- **Input**: Imagens em escala de cinza 256x256x1
- **5 Blocos Convolucionais**:
  - Bloco 1: 32 filtros 5x5 + BatchNorm + MaxPooling 2x2 + Dropout 0.1
  - Bloco 2: 64 filtros 5x5 + BatchNorm + MaxPooling 2x2 + Dropout 0.15
  - Bloco 3: 128 filtros 3x3 + BatchNorm + MaxPooling 2x2 + Dropout 0.2
  - Bloco 4: 256 filtros 3x3 + BatchNorm + MaxPooling 2x2 + Dropout 0.25
  - Bloco 5: 256 filtros 3x3 + BatchNorm + MaxPooling 2x2 + Dropout 0.25
- **Global Average Pooling**: Reduz dimensões e previne overfitting
- **Camadas Densas**: Similar à CNN de pneumonia, com L2 regularization

**Justificativa da arquitetura mais profunda**:
- Mamografias podem requerer análise mais detalhada
- Mais camadas para capturar padrões sutis de lesões
- Global Average Pooling reduz parâmetros e melhora generalização

**3. Configurações de Treinamento**

- **Otimizador**: AdamW (com weight decay) ou Adam
- **Learning Rate**: 0.0001 (reduzido para treinamento mais estável)
- **Loss**: Categorical Crossentropy ou Focal Loss (para classes desbalanceadas)
- **Métricas**: Accuracy, Precision, Recall
- **Batch Size**: 32
- **Épocas**: 50 (com early stopping)
- **Early Stopping**: Patience=10, monitor='val_loss'
- **Model Checkpoint**: Salva melhor modelo baseado em val_loss e val_accuracy
- **ReduceLROnPlateau**: Reduz learning rate quando validação estagna

**4. Callbacks**

1. **ModelCheckpoint**: Salva o melhor modelo durante treinamento
2. **EarlyStopping**: Para treinamento quando não há melhoria
3. **ReduceLROnPlateau**: Ajusta learning rate dinamicamente

### Justificativa da Escolha das Métricas

Em problemas de diagnóstico médico, a escolha das métricas de avaliação é crítica e deve considerar o contexto clínico e os custos associados a diferentes tipos de erro. Neste projeto, utilizamos quatro métricas principais: **Accuracy**, **Precision**, **Recall** e **F1-Score**. A seguir, justificamos a escolha de cada uma:

#### Por que Accuracy não é suficiente?

A **Accuracy** (Acurácia) mede a proporção de predições corretas sobre o total. Embora seja uma métrica intuitiva, ela pode ser enganosa em problemas médicos, especialmente quando há desbalanceamento de classes:

- **Limitação**: Em um dataset com 62% de casos benignos e 38% malignos, um modelo que sempre prediz "benigno" teria 62% de accuracy, mas seria completamente inútil para detectar câncer
- **Uso adequado**: A accuracy é útil como métrica geral, mas não deve ser a única considerada em diagnóstico médico

#### Por que Recall é crítico em diagnóstico médico?

O **Recall** (Sensibilidade) mede a proporção de casos positivos (malignos) que foram corretamente identificados:

- **Importância clínica**: Em diagnóstico de câncer, **falsos negativos são extremamente perigosos** - um caso maligno não detectado pode resultar em progressão da doença e pior prognóstico
- **Interpretação**: Um Recall de 92.86% significa que o modelo detecta 92.86% dos casos malignos, mas ainda falha em detectar 7.14% (3 casos no nosso conjunto de teste)
- **Custo do erro**: O custo de não detectar um câncer maligno é muito maior que o custo de um falso positivo (que pode ser resolvido com exames adicionais)

#### Por que Precision é importante?

A **Precision** (Precisão) mede a proporção de predições positivas que são realmente corretas:

- **Importância clínica**: **Falsos positivos** podem causar ansiedade desnecessária, exames invasivos adicionais (biópsias) e custos médicos
- **Interpretação**: Uma Precision de 100% (Random Forest) significa que quando o modelo prediz "maligno", está sempre correto - não há falsos alarmes
- **Balanceamento**: Alta precision reduz o número de biópsias desnecessárias, mas não deve comprometer o recall

#### Por que F1-Score é uma métrica balanceada?

O **F1-Score** é a média harmônica entre Precision e Recall:

- **Vantagem**: Balanceia a importância de detectar casos positivos (Recall) e evitar falsos alarmes (Precision)
- **Uso**: Útil quando precisamos de uma única métrica que considere ambos os aspectos
- **Limitação**: Assume que Precision e Recall têm igual importância, o que pode não ser verdade em todos os contextos médicos

#### Considerações para o Problema de Câncer de Mama

Para diagnóstico de câncer de mama, a hierarquia de importância das métricas é:

1. **Recall (mais crítico)**: Não perder casos malignos é a prioridade máxima
2. **Precision (importante)**: Evitar alarmes falsos reduz ansiedade e custos
3. **F1-Score**: Fornece uma visão balanceada do desempenho geral
4. **Accuracy**: Útil como métrica geral, mas não suficiente isoladamente

**Conclusão**: A combinação dessas métricas permite uma avaliação completa do modelo, considerando tanto a capacidade de detectar casos críticos quanto a precisão das predições positivas. Em um contexto clínico real, médicos podem ajustar o threshold de decisão baseado na importância relativa de Recall vs Precision para cada paciente específico.

### Resultados e Interpretação

#### Dados Tabulares

**Desempenho dos Modelos**

**Regressão Logística**:
- **Accuracy (Teste)**: 96.49%
- **Precision (M)**: 97.67%
- **Recall (M)**: 92.86%
- **F1-Score (M)**: 95.24%

**Random Forest**:
- **Accuracy (Teste)**: 97.37%
- **Precision (M)**: 100.00%
- **Recall (M)**: 92.86%
- **F1-Score (M)**: 96.30%

**Análise Comparativa**:
O **Random Forest** apresentou desempenho ligeiramente superior em todas as métricas:
- **Accuracy**: 0.88 pontos percentuais a mais
- **Precision**: 2.33 pontos percentuais a mais (100% vs 97.67%)
- **F1-Score**: 1.06 pontos percentuais a mais

Ambos os modelos apresentam desempenho excelente (>95% accuracy), indicando que o problema é relativamente bem separável com as features disponíveis.

**Matriz de Confusão (Random Forest)**:
```
                Predito
              B      M
Real    B    72     0
        M     3    39
```

- **Verdadeiros Negativos (TN)**: 72
- **Falsos Positivos (FP)**: 0
- **Falsos Negativos (FN)**: 3
- **Verdadeiros Positivos (TP)**: 39

**Análise**:
- Nenhum falso positivo: Todos os casos benignos foram corretamente identificados
- 3 falsos negativos: 3 casos malignos foram classificados como benignos
- **Impacto clínico**: Falsos negativos são mais críticos (caso maligno não detectado)

**Feature Importance**:
As features mais importantes identificadas pelo Random Forest foram:

1. `concave points_worst` - Pontos côncavos (pior valor)
2. `perimeter_worst` - Perímetro (pior valor)
3. `concave points_mean` - Pontos côncavos (média)
4. `radius_worst` - Raio (pior valor)
5. `area_worst` - Área (pior valor)

**Interpretação**: Características relacionadas a concavidade e tamanho (perímetro, raio, área) são as mais preditivas, especialmente os valores "worst" (piores), que representam as características mais extremas encontradas.

**Análise SHAP**:
A análise SHAP (SHapley Additive exPlanations) fornece interpretabilidade adicional:

**Insights Globais**:
- Confirma a importância das features identificadas pela feature importance
- Mostra que valores altos de características como `concave points_worst` e `perimeter_worst` aumentam a probabilidade de diagnóstico maligno
- Valores baixos dessas características indicam diagnóstico benigno

**Interpretação Local**:
- Permite entender por que cada predição específica foi feita
- Útil para explicar decisões do modelo a médicos e pacientes
- Mostra a contribuição individual de cada feature para cada caso

#### Dados de Imagens

**Métricas de Avaliação**:
Ambos os modelos de CNN foram avaliados usando:
- **Accuracy**: Taxa de acerto geral
- **Precision**: Precisão por classe
- **Recall**: Sensibilidade por classe
- **F1-Score**: Média harmônica de precision e recall
- **ROC-AUC**: Área sob a curva ROC
- **Matriz de Confusão**: Visualização de erros

**Interpretabilidade: Grad-CAM**:
**Grad-CAM (Gradient-weighted Class Activation Mapping)** foi implementado para visualizar as regiões da imagem que mais influenciam a predição do modelo.

**Como funciona**:
1. Calcula gradientes da classe predita em relação à última camada convolucional
2. Cria um heatmap mostrando regiões importantes
3. Superpõe o heatmap na imagem original

**Benefícios**:
- **Transparência**: Mostra o que o modelo está "vendo"
- **Validação**: Permite verificar se o modelo foca em regiões clinicamente relevantes
- **Debugging**: Identifica se o modelo está aprendendo padrões corretos ou artefatos
- **Confiança**: Ajuda médicos a confiar nas predições do modelo

**Aplicação**:
- Visualização de regiões de atenção para casos de pneumonia
- Identificação de lesões suspeitas em mamografias
- Análise de casos corretos e incorretos

### Discussão Crítica e Limitações

#### Limitações Identificadas

**1. Dataset Limitado**:
- Apenas ~570 amostras podem limitar generalização
- Dataset específico de câncer de mama
- Possível viés geográfico/temporal

**2. Features Disponíveis**:
- Apenas características numéricas de exames
- Não considera histórico médico, genética ou estilo de vida
- Pode não capturar todas as interações relevantes

**3. Desbalanceamento de Classes**:
- Classe benigna tem mais amostras que maligna
- Apesar da estratificação, pode impactar casos raros

**4. Generalização**:
- Modelo treinado em dados históricos
- Não testado em diferentes populações
- Validação externa necessária

**5. Interpretabilidade**:
- Random Forest é mais complexo que modelos lineares
- SHAP ajuda, mas requer conhecimento técnico

#### Viabilidade de Uso Prático

**Pontos Positivos**:
- Alta acurácia (>97%) sugere potencial para triagem inicial
- Modelo rápido e eficiente
- Pode auxiliar na priorização de casos
- Interpretabilidade via SHAP e Grad-CAM

**Considerações Importantes**:
- **NÃO substitui o diagnóstico médico** - deve ser usado apenas como ferramenta de apoio
- Requer validação clínica extensiva
- Necessita integração com sistemas hospitalares
- Treinamento de equipe médica necessário
- Monitoramento contínuo essencial

**Casos de Uso Sugeridos**:
- Triagem inicial para priorização
- Segunda opinião para validação
- Educação médica
- Pesquisa e identificação de padrões
- Controle de qualidade

**Limitações para Uso Clínico**:
- Não deve ser único critério para diagnóstico
- Não considera contexto clínico completo
- Pode gerar falsos positivos/negativos graves
- Requer aprovação regulatória
- Necessita auditoria e responsabilização

### Considerações Éticas e Médicas

**Privacidade e Segurança**:
- Dados médicos sensíveis requerem proteção rigorosa (LGPD, HIPAA)
- Anonimização adequada necessária
- Criptografia e controle de acesso essenciais

**Responsabilidade e Transparência**:
- Responsabilidade final sempre do médico
- Transparência sobre limitações e taxa de erro
- Documentação clara do processo
- Possibilidade de apelação/revisão

**Viés e Equidade**:
- Verificar viés contra grupos demográficos
- Garantir representatividade do dataset
- Monitorar desempenho em subpopulações
- Evitar discriminação

**Impacto no Relacionamento Médico-Paciente**:
- IA não deve substituir comunicação médico-paciente
- Explicações compreensíveis para pacientes
- Respeitar autonomia do paciente
- Manter humanização do cuidado

**Qualidade e Validação**:
- Validação em múltiplos centros
- Comparação com padrão-ouro
- Estudos prospectivos necessários
- Revisão periódica do modelo

**Princípio Fundamental**: O modelo deve sempre servir como **FERRAMENTA DE APOIO** à decisão médica, nunca como substituto do julgamento clínico profissional.

---

## 📈 Resultados Esperados

### Dados Tabulares

#### Regressão Logística

- **Accuracy**: ~96.5%
- **Precision (M)**: ~97.7%
- **Recall (M)**: ~92.9%
- **F1-Score (M)**: ~95.2%

#### Random Forest (Melhor Modelo)

- **Accuracy**: ~97.4%
- **Precision (M)**: ~100.0%
- **Recall (M)**: ~92.9%
- **F1-Score (M)**: ~96.3%

### Classificação de Imagens (CNNs)

#### Pneumonia em Raio-X

- **Modelo**: CNN construída do zero
- **Arquitetura**: 4 blocos convolucionais + camadas densas
- **Input**: Imagens RGB 224x224
- **Métricas Esperadas**: Accuracy > 80% (benchmark para CNNs simples)

#### Câncer de Mama em Mamografias

- **Modelo**: CNN adaptada para escala de cinza
- **Arquitetura**: 5 blocos convolucionais + camadas densas
- **Input**: Imagens em escala de cinza 256x256
- **Métricas Esperadas**: Accuracy > 80%

### Features Mais Importantes (Dados Tabulares)

As características mais preditivas identificadas:

1. `concave points_worst` - Pontos côncavos (pior valor)
2. `perimeter_worst` - Perímetro (pior valor)
3. `concave points_mean` - Pontos côncavos (média)
4. `radius_worst` - Raio (pior valor)
5. `area_worst` - Área (pior valor)

---

## 🔍 Interpretabilidade

### Dados Tabulares

1. **Feature Importance**: Importância global das features (Random Forest)
2. **SHAP Values**:
   - Interpretabilidade local (por predição)
   - Interpretabilidade global (visão geral)
   - Waterfall plots para casos específicos

### Classificação de Imagens

1. **Grad-CAM**: Visualização das regiões da imagem que mais influenciam a predição
   - Heatmaps sobrepostos nas imagens
   - Análise de casos corretos e incorretos
   - Identificação de padrões aprendidos pelo modelo

---

## ⚠️ Limitações e Considerações

### Limitações Técnicas

- Dataset limitado (~570 amostras)
- Apenas características numéricas de exames
- Não considera histórico médico completo
- Possível viés geográfico/temporal

### Considerações para Uso Clínico

- **NÃO substitui o diagnóstico médico**
- Requer validação clínica extensiva
- Necessita aprovação regulatória
- Monitoramento contínuo essencial
- Transparência e responsabilidade ética

Para mais detalhes, consulte a seção de **Discussão Crítica** no notebook `02_tabular_modelagem.ipynb` e o `relatorio_tecnico.md`.

---

## 🐳 Docker (Opcional)

Para executar em container Docker:

```bash
# Construir imagem
docker build -t tech-challenge .

# Executar container
docker run -it -p 8888:8888 tech-challenge
```

---

## 📚 Documentação Adicional

- **Relatório Técnico**: `relatorio_tecnico.md` - Documentação completa do projeto
- **Notebooks**: Contêm análise detalhada e comentários explicativos
- **Código Fonte**: Funções modulares em `src/tabular/` e `src/vision/`

---

## 👥 Contribuição

Este projeto foi desenvolvido como parte do Tech Challenge Fase 1.

---

## 📄 Licença

Consulte o arquivo `LICENSE` para mais informações.

---

## 📞 Contato

Para dúvidas ou sugestões, abra uma issue no repositório.
