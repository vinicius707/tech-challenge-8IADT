# Tech Challenge - Fase 1

## Sistema Inteligente de Suporte ao Diagnóstico Médico

Este projeto implementa modelos de machine learning para **classificação de exames médicos**, utilizando dados estruturados para auxiliar profissionais de saúde na tomada de decisão clínica.

> ⚠️ **IMPORTANTE**: Este sistema não substitui o médico. Ele atua como ferramenta de apoio e triagem. A decisão final sempre deve ser do profissional médico qualificado.

---

## 📌 Problema Abordado

### Classificação de Câncer de Mama

Classificação binária para diagnóstico de **câncer de mama** em duas categorias:

- **B (Benigno)**: Tumor benigno
- **M (Maligno)**: Tumor maligno

O modelo utiliza características clínicas numéricas obtidas de exames médicos (raio, textura, perímetro, área, suavidade, compactação, concavidade, etc.) para fazer predições.

---

## 🧪 Dataset Utilizado

- **Dataset**: Wisconsin Breast Cancer Dataset
- **Fonte**: UCI Machine Learning Repository
- **Tamanho**: 569 amostras
- **Features**: 30 características numéricas
- **Distribuição**: ~62% benigno, ~38% maligno
- **Localização**: `data/tabular/breast-cancer.csv`

### Características do Dataset

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
│   └── tabular/
│       └── breast-cancer.csv          # Dataset principal
├── notebooks/
│   ├── 01_tabular_exploracao.ipynb   # Análise exploratória de dados
│   └── 02_tabular_modelagem.ipynb    # Modelagem e avaliação
├── src/
│   └── tabular/
│       ├── processing.py              # Funções de pré-processamento
│       └── evaluate.py                # Funções de avaliação
├── models/
│   └── maternal_risk_model.pkl        # Modelo treinado salvo
├── config.yaml                        # Configurações do projeto
├── requirements.txt                   # Dependências Python
├── Dockerfile                         # Containerização
├── README.md                          # Este arquivo
└── relatorio_tecnico.md               # Relatório técnico completo
```

---

## 🚀 Instalação e Configuração

### Pré-requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)

### Passo 1: Clonar o Repositório

```bash
git clone <url-do-repositorio>
cd tech-challenge-8IADT
```

### Passo 2: Instalar Dependências

```bash
pip install -r requirements.txt
```

**Principais dependências**:

- `pandas`: Manipulação de dados
- `numpy`: Computação numérica
- `scikit-learn`: Machine learning
- `matplotlib` e `seaborn`: Visualização
- `shap`: Interpretabilidade de modelos
- `jupyter`: Notebooks interativos

### Passo 3: Verificar Dataset

Certifique-se de que o arquivo `data/tabular/breast-cancer.csv` está presente no diretório.

---

## 📊 Uso do Projeto

### Executar Análise Exploratória

1. Abra o Jupyter Notebook:

```bash
jupyter notebook notebooks/01_tabular_exploracao.ipynb
```

2. Execute todas as células para:
   - Carregar e explorar o dataset
   - Visualizar distribuições
   - Analisar correlações entre variáveis

### Executar Modelagem

1. Abra o notebook de modelagem:

```bash
jupyter notebook notebooks/02_tabular_modelagem.ipynb
```

2. Execute todas as células para:
   - Treinar modelos (Regressão Logística e Random Forest)
   - Avaliar desempenho
   - Analisar feature importance e SHAP
   - Salvar o melhor modelo

### Usar Modelo Treinado

```python
from src.tabular.evaluate import load_model, predict
import pandas as pd

# Carregar modelo
model = load_model("models/maternal_risk_model.pkl")

# Preparar dados (exemplo)
# Os dados devem ter as mesmas features usadas no treinamento
new_data = pd.DataFrame({
    'radius_mean': [15.0],
    'texture_mean': [20.0],
    # ... outras features
})

# Fazer predição
prediction = predict(model, new_data)
print(f"Diagnóstico predito: {prediction[0]}")
```

---

## 📈 Resultados Esperados

### Desempenho dos Modelos

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

### Features Mais Importantes

As características mais preditivas identificadas:

1. `concave points_worst` - Pontos côncavos (pior valor)
2. `perimeter_worst` - Perímetro (pior valor)
3. `concave points_mean` - Pontos côncavos (média)
4. `radius_worst` - Raio (pior valor)
5. `area_worst` - Área (pior valor)

---

## 🔍 Interpretabilidade

O projeto implementa duas técnicas de interpretabilidade:

1. **Feature Importance**: Importância global das features (Random Forest)
2. **SHAP Values**:
   - Interpretabilidade local (por predição)
   - Interpretabilidade global (visão geral)
   - Waterfall plots para casos específicos

---

## 📋 Metodologia

### Divisão dos Dados

- **Treino**: 60% (341 amostras)
- **Validação**: 20% (114 amostras)
- **Teste**: 20% (114 amostras)
- **Estratificação**: Mantém proporção de classes

### Pré-processamento

- Remoção de colunas não relevantes
- Tratamento de valores ausentes/infinitos
- Normalização com StandardScaler

### Modelos

- **Regressão Logística**: Baseline interpretável
- **Random Forest**: Modelo ensemble com melhor desempenho

### Métricas de Avaliação

- Accuracy, Precision, Recall, F1-Score
- Matriz de Confusão
- Feature Importance
- SHAP Values

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

## 📚 Documentação Adicional

- **Relatório Técnico**: `relatorio_tecnico.md` - Documentação completa do projeto
- **Notebooks**: Contêm análise detalhada e comentários explicativos
- **Código Fonte**: Funções modulares em `src/tabular/`

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

## 👥 Contribuição

Este projeto foi desenvolvido como parte do Tech Challenge Fase 1.

---

## 📄 Licença

Consulte o arquivo `LICENSE` para mais informações.

---

## 📞 Contato

Para dúvidas ou sugestões, abra uma issue no repositório.

---

**Desenvolvido com ❤️ para auxiliar profissionais de saúde**
