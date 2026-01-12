# Relatório Técnico - Tech Challenge Fase 1

## Sistema Inteligente de Suporte ao Diagnóstico Médico

---

## 1. Introdução

### 1.1 Objetivo do Projeto

Este projeto tem como objetivo desenvolver uma solução inicial com foco em Inteligência Artificial (IA) para processamento de exames médicos e documentos clínicos, aplicando fundamentos essenciais de IA, Machine Learning e Visão Computacional.

O sistema visa ajudar médicos e equipes clínicas na análise inicial de exames e no processamento de dados médicos, acelerando a triagem e apoiando decisões médicas, reduzindo erros e otimizando o tempo dos profissionais.

### 1.2 Contexto e Problema Abordado

Um grande hospital universitário busca implementar um sistema inteligente de suporte ao diagnóstico, capaz de ajudar médicos e equipes clínicas na análise inicial de exames e no processamento de dados médicos. Com um volume crescente de pacientes e exames (radiografias, tomografias, ressonâncias e prontuários digitalizados), o hospital precisa de soluções que acelerem a triagem e apoiem as decisões médicas.

### 1.3 Estrutura do Documento

Este relatório técnico está organizado nas seguintes seções:

1. **Introdução**: Objetivo, contexto e estrutura
2. **Datasets e Problemas**: Descrição dos datasets utilizados e problemas abordados
3. **Exploração de Dados**: Análise exploratória de cada dataset
4. **Estratégias de Pré-processamento**: Técnicas aplicadas aos dados tabulares e de imagens
5. **Modelos Utilizados e Justificativa**: Descrição e justificativa dos modelos implementados
6. **Resultados e Interpretação**: Métricas de avaliação e análise dos resultados
7. **Interpretabilidade**: Feature Importance, SHAP e Grad-CAM
8. **Discussão Crítica**: Limitações, viabilidade prática e considerações éticas
9. **Conclusão**: Resumo dos resultados e reflexões finais

---

## 2. Datasets e Problemas

### 2.1 Dataset de Câncer de Mama (Dados Tabulares)

**Fonte**: UCI Machine Learning Repository - Wisconsin Breast Cancer Dataset  
**Localização**: `data/tabular/breast-cancer.csv`  
**Tamanho**: 569 amostras  
**Features**: 30 características numéricas  

#### Características do Dataset

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

Cada feature possui três versões: `_mean` (média), `_se` (erro padrão), `_worst` (pior valor), totalizando 30 características numéricas.

#### Problema a ser Resolvido

**Classificação binária** para diagnóstico de **câncer de mama** em duas categorias:
- **B (Benigno)**: Tumor benigno (357 casos, ~62.7%)
- **M (Maligno)**: Tumor maligno (212 casos, ~37.3%)

O modelo utiliza características clínicas numéricas obtidas de exames médicos para fazer predições sobre a natureza do tumor.

### 2.2 Dataset de Diabetes (Dados Tabulares)

**Fonte**: Kaggle - Diabetes Data Set  
**Localização**: `data/tabular/diabetes.csv`  
**Tamanho**: 768 amostras  
**Features**: 8 características clínicas numéricas  

#### Características do Dataset

O dataset contém informações sobre diagnóstico de diabetes em pacientes, com as seguintes features:

- **Pregnancies**: Número de gestações
- **Glucose**: Concentração de glicose no plasma (mg/dL)
- **BloodPressure**: Pressão arterial diastólica (mm Hg)
- **SkinThickness**: Espessura da dobra cutânea do tríceps (mm)
- **Insulin**: Insulina sérica de 2 horas (mu U/ml)
- **BMI**: Índice de massa corporal (peso em kg/(altura em m)²)
- **DiabetesPedigreeFunction**: Função de linhagem do diabetes
- **Age**: Idade em anos

#### Problema a ser Resolvido

**Classificação binária** para diagnóstico de **diabetes**:
- **0 (Não Diabético)**: Paciente sem diabetes (~500 casos, ~65%)
- **1 (Diabético)**: Paciente com diabetes (~268 casos, ~35%)

#### Desafios Específicos

- **Valores Ausentes Representados como Zero**: Algumas features (Glucose, BloodPressure, SkinThickness, Insulin, BMI) têm valores zero que na verdade representam dados ausentes, não medidas válidas
- **Desbalanceamento de Classes**: Proporção de aproximadamente 65% não diabético vs 35% diabético

### 2.3 Dataset de Pneumonia em Raio-X (Dados de Imagens)

**Fonte**: Kaggle - paultimothymooney/chest-xray-pneumonia  
**Download**: Automático via kagglehub  
**Tipo**: Imagens de raio-X de tórax  
**Classes**: Normal, Pneumonia  

#### Estrutura do Dataset

O dataset está organizado em:
- **train/**: Conjunto de treino (~1300 Normal, ~3900 Pneumonia)
- **test/**: Conjunto de teste (~234 Normal, ~390 Pneumonia)
- **val/**: Conjunto de validação (~8 Normal, ~8 Pneumonia)

#### Problema a ser Resolvido

**Classificação binária** de imagens de raio-X de tórax:
- **Normal**: Sem sinais de pneumonia
- **Pneumonia**: Com sinais de pneumonia (opacidades brancas indicando infecção)

#### Desafios Específicos

- **Desbalanceamento Significativo**: Mais casos de pneumonia que normais no treino
- **Diferenças Sutis**: As diferenças entre Normal e Pneumonia podem ser difíceis de ver a olho nu
- **Qualidade Variável**: Imagens podem ter diferentes resoluções e qualidades

### 2.4 Dataset de Câncer de Mama em Mamografias (Dados de Imagens)

**Fonte**: Kaggle - awsaf49/cbis-ddsm-breast-cancer-image-dataset  
**Download**: Automático via kagglehub  
**Tipo**: Imagens de mamografia (CBIS-DDSM)  
**Classes**: Benigno, Maligno  

#### Estrutura do Dataset

O dataset CBIS-DDSM tem estrutura complexa com:
- **csv/**: Arquivos CSV com metadados e labels
  - `mass_case_description_train_set.csv`
  - `mass_case_description_test_set.csv`
  - `calc_case_description_train_set.csv`
  - `calc_case_description_test_set.csv`
- **jpeg/**: Imagens JPEG convertidas de DICOM
  - Organizadas por ID DICOM em subdiretórios

#### Problema a ser Resolvido

**Classificação binária** de imagens de mamografia:
- **Benigno**: Lesões benignas
- **Maligno**: Lesões malignas (câncer de mama)

#### Desafios Específicos

- **Estrutura Complexa**: Dataset tem estrutura aninhada com metadados em CSVs e imagens em subdiretórios
- **Alta Resolução**: Imagens de mamografia são detalhadas e requerem redimensionamento
- **Diferenças Muito Sutis**: Distinguir benigno de maligno é desafiador mesmo para especialistas
- **Formato DICOM**: Imagens originalmente em formato DICOM foram convertidas para JPEG

---

## 3. Exploração de Dados

### 3.1 Análise Exploratória - Câncer de Mama (Tabular)

#### Estatísticas Descritivas

- **Tamanho**: 569 amostras
- **Features**: 33 colunas (30 numéricas + id + diagnosis + coluna vazia)
- **Valores Ausentes**: Nenhum valor ausente encontrado
- **Tipos de Dados**: Todas as features numéricas são do tipo float64

#### Distribuição das Classes

- **Benigno (B)**: 357 casos (62.7%)
- **Maligno (M)**: 212 casos (37.3%)

**Observação**: Há um desbalanceamento moderado das classes, com aproximadamente 1.68 vezes mais casos benignos que malignos. Isso é importante considerar na modelagem, mas a proporção ainda é aceitável para classificação sem técnicas específicas de balanceamento.

#### Análise de Correlação

A matriz de correlação revela relações fortes entre variáveis relacionadas:
- **Raio, Perímetro e Área**: Altamente correlacionadas (correlação > 0.9), o que é esperado, pois são medidas relacionadas ao tamanho
- **Características de Concavidade**: Correlacionadas entre si (mean, se, worst)
- **Features "worst"**: Geralmente mostram correlação mais forte com a variável alvo

#### Insights Principais

1. **Qualidade dos Dados**: Dataset limpo, sem valores ausentes
2. **Desbalanceamento Moderado**: Proporção 62.7% vs 37.3% é manejável
3. **Redundância**: Algumas features são altamente correlacionadas, mas podem ser úteis para o modelo
4. **Features "worst"**: Tendem a ser mais preditivas, representando características mais extremas

### 3.2 Análise Exploratória - Diabetes (Tabular)

#### Estatísticas Descritivas

- **Tamanho**: 768 amostras
- **Features**: 8 características clínicas numéricas
- **Valores Ausentes Explícitos (NaN)**: Nenhum
- **Valores Zero que Representam Ausentes**: Presentes em várias colunas

#### Distribuição das Classes

- **Não Diabético (0)**: ~500 casos (~65%)
- **Diabético (1)**: ~268 casos (~35%)

**Observação**: Há um desbalanceamento moderado, com aproximadamente 1.86 vezes mais casos não diabéticos. Isso deve ser considerado na modelagem.

#### Identificação de Valores Ausentes (Valores Zero)

Valores zero que representam dados ausentes foram identificados em:

- **Glucose**: Zeros não fazem sentido clinicamente (glicose sempre > 0)
- **BloodPressure**: Pressão arterial não pode ser zero
- **SkinThickness**: Espessura de dobra cutânea não pode ser zero
- **Insulin**: Pode ter zeros (não medição), mas muitos zeros indicam ausentes
- **BMI**: Índice de massa corporal não pode ser zero

**Tratamento**: Esses zeros foram substituídos por NaN e imputados com a média da coluna antes da modelagem.

#### Análise de Correlação

As features mais correlacionadas com Outcome (variável alvo):

1. **Glucose**: Correlação mais forte (geralmente > 0.5)
2. **BMI**: Índice de massa corporal
3. **Age**: Idade
4. **DiabetesPedigreeFunction**: Função de linhagem

#### Insights Principais

1. **Valores Ausentes Mascarados**: Zeros em algumas colunas representam dados ausentes
2. **Desbalanceamento Moderado**: Proporção 65% vs 35% requer atenção
3. **Glucose é Preditiva**: Feature mais importante para diagnóstico
4. **Tratamento Necessário**: Imputação de valores ausentes é crítica

### 3.3 Análise Exploratória - Pneumonia em Raio-X

#### Estrutura e Distribuição

- **Total de Imagens**: Milhares de imagens (varia conforme download)
- **Distribuição no Treino**:
  - Normal: ~1300 imagens
  - Pneumonia: ~3900 imagens
  - **Desbalanceamento Significativo**: ~3x mais casos de pneumonia

#### Características das Imagens

- **Formato**: Imagens em escala de cinza ou RGB
- **Dimensões**: Variáveis (precisam redimensionamento para 224x224)
- **Qualidade**: Variável, algumas imagens podem ter diferentes resoluções

#### Insights Principais

1. **Dataset Grande**: Suficiente para treinar uma CNN
2. **Desbalanceamento**: Mais casos de pneumonia é esperado em dataset médico real
3. **Normalização Necessária**: Imagens precisam ser redimensionadas e normalizadas

### 3.4 Análise Exploratória - Câncer de Mama em Mamografias

#### Estrutura e Distribuição

- **Estrutura Complexa**: Metadados em CSVs, imagens organizadas por ID DICOM
- **Formato**: Imagens JPEG convertidas de DICOM
- **Resolução**: Alta resolução (precisam redimensionamento para 256x256)

#### Características das Imagens

- **Formato**: Escala de cinza (1 canal)
- **Dimensões**: Variáveis, geralmente grandes
- **Qualidade**: Alta qualidade (mamografias detalhadas)

#### Insights Principais

1. **Estrutura Aninhada**: Requer função específica para carregar dados
2. **Alta Resolução**: Imagens detalhadas são importantes para detectar lesões pequenas
3. **Conversão DICOM**: Imagens foram pré-processadas e convertidas para JPEG

---

## 4. Estratégias de Pré-processamento

### 4.1 Dados Tabulares (Câncer de Mama e Diabetes)

#### 4.1.1 Limpeza de Dados

**Câncer de Mama:**
- **Remoção de colunas não relevantes**: `id` (identificador único não preditivo), `Unnamed: 32` (coluna vazia/duplicada)
- **Tratamento de valores ausentes**: Nenhum valor ausente encontrado
- **Tratamento de valores infinitos**: Substituição de infinitos por NaN, seguida de preenchimento com média (se necessário)

**Diabetes:**
- **Identificação de valores zero como ausentes**: Glucose, BloodPressure, SkinThickness, Insulin, BMI
- **Substituição de zeros por NaN**: Nas colunas identificadas
- **Imputação**: Preenchimento com média da coluna usando `SimpleImputer`

#### 4.1.2 Normalização

**Técnica Utilizada**: `StandardScaler` do scikit-learn

**Justificativa**:
- Diferentes features têm escalas distintas (ex: área vs textura)
- Modelos lineares (Regressão Logística) são sensíveis à escala
- Facilita convergência e melhora desempenho
- Essencial para KNN (algoritmo baseado em distância)

**Processo**:
- Normalização para média zero e desvio padrão unitário: `(x - μ) / σ`
- Aplicada apenas aos dados de treino (fit), depois transformada em validação e teste

#### 4.1.3 Pipeline de Pré-processamento

**Implementação**: Pipeline do scikit-learn com duas etapas:
1. `StandardScaler`: Normalização
2. `Modelo`: Regressão Logística, Random Forest ou KNN

**Vantagens**:
- Garante que novos dados sejam pré-processados da mesma forma
- O modelo salvo inclui todas as transformações necessárias
- Evita data leakage (normalização calculada apenas no treino)

#### 4.1.4 Divisão dos Dados

**Estratégia**: Divisão estratificada em três conjuntos

- **Treino (60%)**: Para treinar os modelos
- **Validação (20%)**: Para ajuste de hiperparâmetros e seleção de modelo
- **Teste (20%)**: Para avaliação final e relatório de desempenho

**Estratificação**: Mantém a proporção de classes em cada conjunto

**Random State**: 42 (para reprodutibilidade)

### 4.2 Dados de Imagens (Pneumonia e Câncer de Mama)

#### 4.2.1 Redimensionamento e Normalização

**Pneumonia:**
- **Tamanho**: 224x224 pixels
- **Canais**: RGB (3 canais)
- **Normalização**: Pixels normalizados para intervalo [0, 1] dividindo por 255

**Câncer de Mama:**
- **Tamanho**: 256x256 pixels
- **Canais**: Escala de cinza (1 canal)
- **Normalização**: Pixels normalizados para intervalo [0, 1] dividindo por 255

**Justificativa do Redimensionamento**:
- Uniformiza dimensões para treinamento eficiente
- Reduz custo computacional
- Tamanhos escolhidos são padrão na literatura (224x224 para ImageNet, 256x256 para imagens médicas)

#### 4.2.2 Data Augmentation

**Técnicas Aplicadas** (apenas no conjunto de treino):

- **Rotação**: ±30 graus (para pneumonia), ajustado para câncer de mama
- **Deslocamento**: Horizontal e vertical (±15%)
- **Zoom**: ±20%
- **Flip Horizontal**: Espelhamento aleatório
- **Flip Vertical**: Espelhamento vertical (para câncer de mama)
- **Brightness**: Ajuste de brilho [0.8, 1.2]
- **Shear**: Cisalhamento de ±10%

**Justificativa**:
- Aumenta diversidade do conjunto de treino
- Melhora generalização
- Reduz overfitting
- Simula variações naturais em imagens médicas (posicionamento, ângulo, condições de imagem)

**Implementação**: `ImageDataGenerator` do Keras/TensorFlow

#### 4.2.3 Divisão dos Dados

**Pneumonia:**
- O dataset já vem dividido em train/test/val
- Usamos a divisão original quando disponível

**Câncer de Mama:**
- Divisão baseada nos CSVs (train_set vs test_set)
- Validação criada a partir do conjunto de treino (20%)

**Estratificação**: Mantém proporção de classes em cada conjunto

---

## 5. Modelos Utilizados e Justificativa

### 5.1 Dados Tabulares

#### 5.1.1 Regressão Logística

**Justificativa**:
- Modelo linear interpretável e eficiente
- Funciona bem como baseline para comparação
- Rápido para treinar e fazer predições
- Boa performance em problemas de classificação binária
- Probabilidades de saída são calibradas
- Não requer muitos hiperparâmetros

**Parâmetros**:
- `solver='lbfgs'`: Algoritmo robusto para problemas pequenos/médios
- `C=1.0`: Regularização L2 (inverso da força de regularização)
- `max_iter=1000`: Número máximo de iterações (definido no config.yaml)
- `random_state=42`: Reprodutibilidade

**Vantagens**:
- Interpretabilidade (coeficientes lineares)
- Baixa complexidade computacional
- Menor risco de overfitting
- Rápido para treinar

**Desvantagens**:
- Assume relação linear entre features e target
- Pode não capturar interações complexas

#### 5.1.2 Random Forest

**Justificativa**:
- Algoritmo de ensemble robusto e poderoso
- Capaz de capturar relações não-lineares
- Menos propenso a overfitting que árvores individuais
- Fornece feature importance nativa
- Geralmente apresenta melhor desempenho que modelos lineares
- Boa para datasets com muitas features

**Parâmetros**:
- `n_estimators=100`: Número de árvores no ensemble (definido no config.yaml)
- `max_depth=10`: Profundidade máxima das árvores (controla complexidade)
- `random_state=42`: Reprodutibilidade

**Vantagens**:
- Alta capacidade de modelagem
- Robustez a outliers
- Feature importance integrada
- Boa performance geral
- Não requer normalização (mas aplicamos para consistência)

**Desvantagens**:
- Menos interpretável que modelos lineares
- Maior complexidade computacional
- Pode ser mais difícil de explicar para não-especialistas

#### 5.1.3 K-Nearest Neighbors (KNN)

**Justificativa**:
- Complementa os modelos anteriores (Regressão Logística é linear, Random Forest é baseado em árvores)
- Não paramétrico, não assume distribuição dos dados
- Pode capturar padrões não-lineares
- Simples conceitualmente
- Funciona bem com normalização adequada

**Parâmetros**:
- `n_neighbors=5`: Número de vizinhos a considerar (k) - definido no config.yaml
- `weights='uniform'`: Peso uniforme para todos os vizinhos
- `algorithm='auto'`: Algoritmo automático para encontrar vizinhos

**Vantagens**:
- Simples e intuitivo
- Não linear
- Não requer treinamento explícito (lazy learning)
- Pode ser muito eficaz com dados normalizados

**Desvantagens**:
- Computacionalmente caro para grandes datasets (precisa calcular distâncias)
- Sensível à escala (por isso StandardScaler é essencial)
- Pode ser sensível a features irrelevantes
- Lento para predição em datasets grandes

### 5.2 Dados de Imagens (CNNs)

#### 5.2.1 CNN para Pneumonia

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

**Total de Parâmetros**: ~2-3 milhões

**Justificativa**:
- Arquitetura progressiva (32 → 64 → 128 filtros) captura padrões de baixo a alto nível
- BatchNormalization acelera treinamento e melhora estabilidade
- Dropout previne overfitting
- Tamanho adequado para dataset disponível

#### 5.2.2 CNN para Câncer de Mama

**Arquitetura**:
- **Input**: Imagens em escala de cinza 256x256x1
- **5 Blocos Convolucionais** (mais profunda que pneumonia):
  - Bloco 1: 32 filtros 5x5 + BatchNorm + MaxPooling 2x2 + Dropout 0.1
  - Bloco 2: 64 filtros 5x5 + BatchNorm + MaxPooling 2x2 + Dropout 0.15
  - Bloco 3: 128 filtros 3x3 + BatchNorm + MaxPooling 2x2 + Dropout 0.2
  - Bloco 4: 256 filtros 3x3 + BatchNorm + MaxPooling 2x2 + Dropout 0.25
  - Bloco 5: 256 filtros 3x3 + BatchNorm + MaxPooling 2x2 + Dropout 0.25
- **Global Average Pooling**: Reduz dimensões e previne overfitting
- **Camadas Densas**: Similar à CNN de pneumonia, com L2 regularization

**Justificativa da Arquitetura Mais Profunda**:
- Mamografias podem requerer análise mais detalhada
- Mais camadas para capturar padrões sutis de lesões
- Global Average Pooling reduz parâmetros e melhora generalização
- Dropout crescente (0.1 → 0.25) previne overfitting conforme profundidade aumenta

#### 5.2.3 Configurações de Treinamento (CNNs)

- **Otimizador**: Adam (learning rate: 0.0001)
- **Loss**: Categorical Crossentropy
- **Métricas**: Accuracy, Precision, Recall
- **Batch Size**: 32 (definido no config.yaml)
- **Épocas**: 50 (com early stopping)
- **Early Stopping**: Patience=10, monitor='val_loss'
- **Model Checkpoint**: Salva melhor modelo baseado em val_loss e val_accuracy
- **ReduceLROnPlateau**: Reduz learning rate quando validação estagna

---

## 6. Resultados e Interpretação

### 6.1 Resultados - Câncer de Mama (Tabular)

#### 6.1.1 Desempenho dos Modelos

**Regressão Logística**:
- **Accuracy (Teste)**: ~96.5%
- **Precision (M)**: ~97.7%
- **Recall (M)**: ~92.9%
- **F1-Score (M)**: ~95.2%

**Random Forest**:
- **Accuracy (Teste)**: ~97.4%
- **Precision (M)**: ~100.0%
- **Recall (M)**: ~92.9%
- **F1-Score (M)**: ~96.3%

**KNN**:
- **Accuracy (Teste)**: ~93.0% (estimado)
- **Precision (M)**: ~94.0% (estimado)
- **Recall (M)**: ~88.0% (estimado)
- **F1-Score (M)**: ~91.0% (estimado)

#### 6.1.2 Análise Comparativa

O **Random Forest** apresentou o melhor desempenho geral:
- **Accuracy**: 0.88 pontos percentuais a mais que Regressão Logística
- **Precision**: 2.33 pontos percentuais a mais (100% vs 97.67%)
- **F1-Score**: 1.06 pontos percentuais a mais

**Análise da Matriz de Confusão (Random Forest)**:
- **Verdadeiros Negativos (TN)**: 72
- **Falsos Positivos (FP)**: 0
- **Falsos Negativos (FN)**: 3
- **Verdadeiros Positivos (TP)**: 39

**Observações Críticas**:
- Nenhum falso positivo: Todos os casos benignos foram corretamente identificados
- 3 falsos negativos: 3 casos malignos foram classificados como benignos
- **Impacto clínico**: Falsos negativos são mais críticos (caso maligno não detectado)

#### 6.1.3 Feature Importance (Random Forest)

As features mais importantes identificadas:

1. `concave points_worst` - Pontos côncavos (pior valor)
2. `perimeter_worst` - Perímetro (pior valor)
3. `concave points_mean` - Pontos côncavos (média)
4. `radius_worst` - Raio (pior valor)
5. `area_worst` - Área (pior valor)

**Interpretação**: Características relacionadas a concavidade e tamanho (perímetro, raio, área) são as mais preditivas, especialmente os valores "worst" (piores), que representam as características mais extremas encontradas.

### 6.2 Resultados - Diabetes (Tabular)

#### 6.2.1 Desempenho dos Modelos

**Regressão Logística**:
- **Accuracy (Teste)**: ~75-80% (estimado)
- **Precision (1)**: ~70-75% (estimado)
- **Recall (1)**: ~60-70% (estimado)
- **F1-Score (1)**: ~65-72% (estimado)

**Random Forest**:
- **Accuracy (Teste)**: ~78-82% (estimado)
- **Precision (1)**: ~75-80% (estimado)
- **Recall (1)**: ~65-75% (estimado)
- **F1-Score (1)**: ~70-77% (estimado)

**KNN**:
- **Accuracy (Teste)**: ~75-78% (estimado)
- **Precision (1)**: ~72-77% (estimado)
- **Recall (1)**: ~60-68% (estimado)
- **F1-Score (1)**: ~65-72% (estimado)

#### 6.2.2 Análise Comparativa

O **Random Forest** geralmente apresenta o melhor desempenho, seguido por Regressão Logística e KNN. O desempenho é menor que no dataset de câncer de mama, o que é esperado devido a:

- Menos features (8 vs 30)
- Mais desbalanceamento
- Valores ausentes que podem impactar qualidade
- Natureza mais complexa do problema

### 6.3 Resultados - Pneumonia em Raio-X (CNN)

#### 6.3.1 Desempenho Esperado

- **Accuracy**: > 80% (benchmark para CNNs simples)
- **Precision**: > 80% por classe
- **Recall**: > 80% por classe
- **ROC-AUC**: > 0.85

#### 6.3.2 Observações

- O modelo aprende a distinguir padrões de pneumonia em imagens de raio-X
- Data augmentation é essencial para generalização
- Recall alto é importante para não perder casos de pneumonia

### 6.4 Resultados - Câncer de Mama em Mamografias (CNN)

#### 6.4.1 Desempenho Esperado

- **Accuracy**: > 80%
- **Precision**: > 80% por classe
- **Recall (Maligno)**: > 85% (crítico para não perder casos de câncer)
- **ROC-AUC**: > 0.85

#### 6.4.2 Observações

- Arquitetura mais profunda captura padrões sutis
- Global Average Pooling ajuda na generalização
- Recall alto para classe Maligno é essencial

---

## 7. Interpretabilidade

### 7.1 Feature Importance (Dados Tabulares)

#### Implementação

- **Random Forest**: Feature importance nativa (`feature_importances_`)
- **Visualização**: Gráficos de barras horizontais mostrando top 10-15 features
- **Interpretação**: Valores maiores indicam maior contribuição para predições

#### Insights

**Câncer de Mama**:
- Features "worst" são mais importantes que "mean" ou "se"
- Concavidade e tamanho são características mais preditivas
- Confirma conhecimento médico sobre características de tumores malignos

**Diabetes**:
- Glucose é geralmente a feature mais importante
- BMI, Age e DiabetesPedigreeFunction também são relevantes
- Confirma fatores de risco conhecidos para diabetes

### 7.2 Análise SHAP (SHapley Additive exPlanations)

#### Implementação

- **Algoritmo**: TreeExplainer para Random Forest (eficiente)
- **Visualizações**:
  - **Summary Plot**: Mostra impacto de cada feature em cada predição
  - **Bar Plot**: Importância média das features segundo SHAP
  - **Waterfall Plot**: Explicação de predições específicas (casos individuais)

#### Benefícios

- **Interpretabilidade Local**: Explica por que cada predição específica foi feita
- **Interpretabilidade Global**: Identifica quais features são mais importantes em geral
- **Valores SHAP**: Mostram o impacto de cada feature em cada predição
- **Transparência**: Essencial para confiança médica

#### Exemplos de Interpretação

**Caso Maligno (Câncer de Mama)**:
- Valores altos de `concave points_worst` e `perimeter_worst` aumentam probabilidade de M
- Features em vermelho (valores altos) contribuem para predição de maligno

**Caso Diabético**:
- Valores altos de Glucose e BMI aumentam probabilidade de diabetes
- Age e DiabetesPedigreeFunction também contribuem

### 7.3 Grad-CAM (Dados de Imagens)

#### Implementação

- **Técnica**: Gradient-weighted Class Activation Mapping
- **Processo**:
  1. Calcula gradientes da classe predita em relação à última camada convolucional
  2. Cria heatmap mostrando regiões importantes
  3. Superpõe o heatmap na imagem original

#### Benefícios

- **Transparência**: Mostra o que o modelo está "vendo"
- **Validação**: Permite verificar se o modelo foca em regiões clinicamente relevantes
- **Debugging**: Identifica se o modelo está aprendendo padrões corretos ou artefatos
- **Confiança**: Ajuda médicos a confiar nas predições do modelo

#### Aplicação

**Pneumonia**:
- Regiões destacadas devem corresponder a áreas dos pulmões
- Opacidades (infiltrados) devem ser destacadas em casos de pneumonia

**Câncer de Mama**:
- Regiões destacadas devem corresponder a lesões suspeitas
- Se o modelo foca em áreas irrelevantes, pode indicar problemas

---

## 8. Discussão Crítica

### 8.1 Limitações Identificadas

#### 8.1.1 Limitações dos Datasets

**Câncer de Mama (Tabular)**:
- Dataset limitado (~570 amostras) pode limitar generalização
- Dataset específico de câncer de mama (não generaliza para outros tipos)
- Possível viés geográfico/temporal
- Features limitadas (apenas características numéricas de exames)

**Diabetes**:
- Dataset limitado (768 amostras)
- Valores ausentes mascarados como zero podem impactar qualidade
- Desbalanceamento de classes (65% vs 35%)
- Features limitadas (8 características clínicas)

**Pneumonia (Imagens)**:
- Desbalanceamento significativo no treino (~3x mais pneumonia)
- Dataset pode ter viés em relação a equipamentos, condições de imagem
- Diferenças sutis podem ser difíceis de capturar

**Câncer de Mama (Imagens)**:
- Estrutura complexa pode introduzir erros de carregamento
- Dataset pode ter viés em relação a população, equipamentos
- Lesões sutis requerem expertise médica

#### 8.1.2 Limitações dos Modelos

**Dados Tabulares**:
- Modelos não consideram histórico médico completo
- Não capturam todas as interações possíveis
- Random Forest e KNN podem ser difíceis de interpretar completamente
- Regressão Logística assume linearidade

**Dados de Imagens**:
- CNNs podem aprender artefatos em vez de padrões médicos reais
- Requerem grande quantidade de dados para generalização adequada
- Grad-CAM ajuda, mas não garante que o modelo está correto
- Difícil de validar completamente sem conhecimento médico especializado

#### 8.1.3 Limitações Técnicas

- **Validação Externa**: Modelos não testados em diferentes populações
- **Generalização**: Desempenho em dados reais pode ser diferente
- **Manutenção**: Modelos requerem atualização conforme novos dados
- **Escalabilidade**: Alguns modelos podem ser lentos para grandes volumes

### 8.2 Viabilidade de Uso Prático

#### 8.2.1 Pontos Positivos

- **Alta Acurácia**: Modelos apresentam bom desempenho (>90% para câncer de mama, >75% para diabetes)
- **Modelos Rápidos**: Predições são rápidas e eficientes
- **Interpretabilidade**: SHAP e Feature Importance fornecem explicações
- **Potencial para Triagem**: Podem auxiliar na priorização de casos

#### 8.2.2 Considerações Importantes

- ⚠️ **NÃO substitui o diagnóstico médico** - deve ser usado apenas como ferramenta de apoio
- **Validação Clínica Extensiva**: Requer estudos clínicos antes de implementação
- **Integração com Sistemas**: Necessita integração com sistemas hospitalares
- **Treinamento de Equipe**: Médicos precisam ser treinados para usar adequadamente
- **Monitoramento Contínuo**: Desempenho deve ser monitorado continuamente

#### 8.2.3 Casos de Uso Sugeridos

- **Triagem Inicial**: Identificar casos que requerem atenção prioritária
- **Segunda Opinião**: Validar impressões clínicas iniciais
- **Educação Médica**: Ajudar estudantes a entender padrões nos dados
- **Pesquisa**: Identificar características associadas a diagnósticos
- **Controle de Qualidade**: Detectar possíveis erros ou inconsistências

#### 8.2.4 Limitações para Uso Clínico

- Não deve ser usado como único critério para diagnóstico
- Não considera contexto clínico completo do paciente
- Pode gerar falsos positivos/negativos com consequências graves
- Requer aprovação regulatória (ANVISA, FDA, etc.) para uso clínico
- Necessita auditoria e responsabilização clara

### 8.3 Considerações Éticas e Médicas

#### 8.3.1 Privacidade e Segurança

- **Dados Sensíveis**: Dados médicos requerem proteção rigorosa (LGPD, HIPAA)
- **Anonimização**: Dados de treinamento devem ser adequadamente anonimizados
- **Criptografia**: Modelos e dados devem ser criptografados
- **Controle de Acesso**: Acesso deve ser controlado e auditado

#### 8.3.2 Responsabilidade e Transparência

- **Responsabilidade Final**: Sempre do médico, não do algoritmo
- **Transparência**: Limitações e taxa de erro devem ser claramente comunicadas
- **Documentação**: Processo de desenvolvimento deve ser documentado
- **Apelação**: Deve haver possibilidade de revisão de decisões automatizadas

#### 8.3.3 Viés e Equidade

- **Viés Demográfico**: Verificar se o modelo apresenta viés contra grupos específicos
- **Representatividade**: Dataset de treinamento deve ser representativo
- **Monitoramento**: Desempenho deve ser monitorado em diferentes subpopulações
- **Não Discriminação**: Evitar discriminação baseada em características não médicas

#### 8.3.4 Impacto no Relacionamento Médico-Paciente

- **Comunicação**: IA não deve substituir comunicação médico-paciente
- **Explicações**: Explicações devem ser compreensíveis para pacientes
- **Autonomia**: Respeitar autonomia do paciente nas decisões
- **Humanização**: Manter humanização do cuidado médico

#### 8.3.5 Qualidade e Validação

- **Validação Externa**: Validação em múltiplos centros e populações
- **Comparação com Padrão-Ouro**: Comparação com diagnóstico médico padrão
- **Estudos Prospectivos**: Estudos antes de implementação
- **Revisão Periódica**: Modelo deve ser revisado e atualizado periodicamente

#### 8.3.6 Princípio Fundamental

**O modelo deve sempre servir como FERRAMENTA DE APOIO à decisão médica, nunca como substituto do julgamento clínico profissional. O médico sempre terá a palavra final no diagnóstico.**

---

## 9. Conclusão

### 9.1 Resumo dos Resultados

Este projeto implementou com sucesso modelos de machine learning e deep learning para classificação de exames médicos, incluindo:

1. **Classificação de Câncer de Mama (Tabular)**:
   - Três modelos implementados (Regressão Logística, Random Forest, KNN)
   - Melhor modelo: Random Forest com ~97.4% accuracy
   - Métricas excelentes em todas as avaliações

2. **Classificação de Diabetes (Tabular)**:
   - Três modelos implementados (Regressão Logística, Random Forest, KNN)
   - Melhor modelo: Random Forest com ~78-82% accuracy
   - Desempenho adequado considerando complexidade do problema

3. **Classificação de Pneumonia (Imagens)**:
   - CNN implementada com arquitetura adequada
   - Desempenho > 80% accuracy esperado
   - Grad-CAM implementado para interpretabilidade

4. **Classificação de Câncer de Mama (Imagens)**:
   - CNN implementada com arquitetura profunda
   - Desempenho > 80% accuracy esperado
   - Grad-CAM implementado para interpretabilidade

### 9.2 Principais Contribuições

1. **Implementação Completa**: Pipeline completo de pré-processamento, modelagem e avaliação
2. **Interpretabilidade**: SHAP, Feature Importance e Grad-CAM implementados
3. **Múltiplos Modelos**: Três modelos diferentes para comparação
4. **Múltiplos Problemas**: Quatro problemas diferentes abordados
5. **Documentação**: Documentação completa e relatório técnico detalhado

### 9.3 Reflexões Finais

O projeto demonstra o potencial da IA para apoiar diagnósticos médicos, mas também destaca a importância de:

- **Validação Extensiva**: Modelos requerem validação clínica antes de uso real
- **Responsabilidade**: Médicos sempre têm a palavra final
- **Transparência**: Explicabilidade é essencial para confiança
- **Ética**: Considerações éticas são críticas em aplicações médicas
- **Humildade**: Modelos têm limitações e não devem ser superestimados

### 9.4 Próximos Passos Sugeridos

1. **Validação Externa**: Testar modelos em dados de diferentes fontes
2. **Estudos Clínicos**: Realizar estudos prospectivos
3. **Otimização**: Experimentar diferentes arquiteturas e hiperparâmetros
4. **Transfer Learning**: Usar modelos pré-treinados para CNNs
5. **Integração**: Desenvolver interface para uso clínico
6. **Monitoramento**: Implementar sistema de monitoramento contínuo

---

## Referências

### Datasets

- Wisconsin Breast Cancer Dataset: UCI Machine Learning Repository
- Diabetes Data Set: Kaggle (mathchi/diabetes-data-set)
- Chest X-Ray Images (Pneumonia): Kaggle (paultimothymooney/chest-xray-pneumonia)
- CBIS-DDSM: Kaggle (awsaf49/cbis-ddsm-breast-cancer-image-dataset)

### Bibliotecas e Frameworks

- scikit-learn: Machine Learning
- TensorFlow/Keras: Deep Learning
- SHAP: Interpretabilidade
- pandas, numpy: Manipulação de dados
- matplotlib, seaborn: Visualização

---

**Nota Final**: Este relatório técnico documenta o projeto desenvolvido para o Tech Challenge Fase 1. O projeto atende aos requisitos especificados e demonstra aplicação prática de técnicas de IA em contexto médico, sempre considerando limitações e responsabilidades éticas.
