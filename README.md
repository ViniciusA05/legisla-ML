# LegislaML: Scoring e Modelagem de Qualidade de Leis Federais

O **LegislaML** é uma plataforma de Inteligência Artificial voltada para a avaliação, o monitoramento e a predição da qualidade textual de leis federais brasileiras (leis ordinárias e complementares). 

O projeto processa o texto integral das leis publicadas no período de 2020 a 2026 e utiliza processamento de linguagem natural (NLP) e algoritmos de Machine Learning para estimar a probabilidade de uma nova redação sofrer vetos presidenciais ou exigir reformas (instabilidades) futuras devido a problemas formais na escrita jurídica.

---

## Arquitetura do Repositório

```text
legisla-ml/
├── data/
│   ├── raw/              # JSONs de metadados do Senado e textos originais do Sigen
│   └── processed/        # Base limpa de NLP e matriz de features numéricas
├── models/               # Modelos classificadores finais treinados (.joblib)
├── notebooks/
│   └── analysis.ipynb    # Jupyter Notebook com gráficos de EDA, importâncias e matrizes
├── src/
│   ├── scraper.py        # Coletor automatizado da API de Dados Abertos e Sigen
│   ├── preprocessing.py  # Limpeza, tokenização e lematização morfossintática
│   ├── features.py       # Extração de features linguísticas e cálculo do Score
│   └── model.py          # Treinamento e avaliação de modelos de classificação
├── requirements.txt      # Dependências do Python
├── LICENSE               # Licença MIT
└── README.md             # Documentação técnica do repositório
```

---

## Como Executar o Pipeline

### 1. Preparar o Ambiente
Instale os pacotes necessários listados no arquivo de dependências:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Coleta dos Dados
Para baixar a lista de normas do Senado Federal e raspar o texto integral do sistema Sigen (anos 2020 a 2026), execute:
```bash
python src/scraper.py
```
*Esse script gera os arquivos de metadados em JSON e os textos limpos da div `#conteudoPrincipal` na pasta `data/raw/`.*

### 3. Pré-processamento e Processamento Linguístico (NLP)
Para executar a limpeza das leis, tokenização de sentenças e lematização para o português (usando SpaCy):
```bash
python src/preprocessing.py
```
*Gera o arquivo `data/processed/leis_clean.csv` contendo as sentenças e tokens lematizados.*

### 4. Engenharia de Features e Cálculo do Score
Para calcular as métricas linguísticas, gerar o Score analítico (0 a 10) e extrair os rótulos reais de vetos/alterações das leis:
```bash
python src/features.py
```
*Gera a matriz numérica `data/processed/leis_features.csv`.*

### 5. Treinamento do Modelo de Machine Learning
Para treinar os classificadores Random Forest, SVM e XGBoost sobre as 420 leis coletadas:
```bash
python src/model.py
```
*Treina as pipelines com `StandardScaler`, avalia com validação cruzada 5-fold e exporta os melhores modelos finais na pasta `models/`.*

---

## Detalhes Técnicos e Engenharia de Features

Extraímos 6 features textuais e estruturais das leis para descrever a sua qualidade técnica e clareza de redação:
1.  **Legibilidade Flesch (PT):** Mapeia a complexidade sintática (relação sílabas/palavra e palavras/frase).
2.  **Média de Palavras por Sentença:** Identifica prolixidade e frases excessivamente longas.
3.  **Sentenças por Artigo:** Mede a densidade do texto dentro dos artigos legais.
4.  **Densidade de Referências:** Contagem de remissões internas e legislativas (*"art."*, *"§"*, *"inciso"*).
5.  **Type-Token Ratio (TTR):** Razão de diversidade lexical (palavras únicas / total de palavras).
6.  **Proporção de Termos Jurídicos:** Frequência de palavras comuns ao jargão técnico do direito.

### Geração das Variáveis Alvo (Targets Reais)
Para que os algoritmos de inteligência artificial aprendam a prever problemas futuros, extraímos rótulos reais da chave `vides` nos históricos do Senado:
*   `teve_veto` (0 ou 1): Se a lei sofreu mensagem de veto parcial ou total pelo presidente no diário oficial.
*   `instavel` (0 ou 1): Se a lei exigiu reformas, alterações ou revogações permanentes subsequentes devido a ambiguidades ou má formulação.

---

## Resultados Obtidos

Os modelos foram avaliados em um conjunto de teste (20% das leis) mantido oculto durante o treinamento, com dados estratificados para lidar com o desbalanceamento das classes:

### Risco de Veto (`teve_veto`)
*   **Melhor Modelo:** **Random Forest** (Acurácia: **97.6%** | F1-Score Macro: **0.969**)
*   **Performance:** Alcançou **100% de Recall** para leis vetadas no conjunto de teste, garantindo que o classificador identificasse todas as leis com risco de veto real.

### Instabilidade e Reformas (`instavel`)
*   **Melhor Modelo:** **Random Forest** (Acurácia: **92.8%** | F1-Score Macro: **0.897**)
*   **Performance:** Alcançou **80% de Recall** e **88.8% de Precisão** para identificar leis que sofreriam emendas ou revogações posteriores.

*Consulte as análises completas, os gráficos de importância de features e estudos de caso de leis reais abrindo o notebook [notebooks/analysis.ipynb](notebooks/analysis.ipynb).*
