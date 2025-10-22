# 🚢 Previsão de Sobrevivência no Titanic (Kaggle)

Este projeto implementa um pipeline completo de Machine Learning para prever a sobrevivência dos passageiros do Titanic, conforme o desafio clássico do Kaggle. O objetivo é aplicar técnicas de engenharia de features (feature engineering) e encontrar os melhores hiperparâmetros para um modelo `RandomForestClassifier`.

O script realiza todo o processo: carregamento, limpeza, transformação, engenharia de features, busca de hiperparâmetros (simplificada) e treinamento final para gerar o arquivo de submissão.

## 📋 Índice

- [Visão Geral do Projeto](#-visão-geral-do-projeto)
- [Funcionalidades](#-funcionalidades)
- [Metodologia do Pipeline de ML](#-metodologia-do-pipeline-de-ml)
- [Tecnologias Utilizadas](#-tecnologias-utilizadas)
- [Pré-requisitos](#-pré-requisitos)
- [Como Executar](#-como-executar)
- [Estrutura do Código](#-estrutura-do-código)
- [Saídas Geradas](#-saídas-geradas)
- [Autor](#-autor)

## 🎯 Visão Geral do Projeto

O script simula o fluxo de trabalho de um cientista de dados para resolver o desafio do Titanic. Ele lê os dados brutos de treino e teste, aplica uma série de transformações e cria novas features relevantes (como `FamilySize`, `Title`, `Deck`, etc.) para melhorar o poder preditivo do modelo.

O diferencial deste script é a função `preprocess` unificada, que garante que tanto os dados de treino quanto os de teste passem exatamente pelas mesmas transformações, evitando vazamento de dados (data leakage) e inconsistências. Além disso, ele implementa uma busca simples por hiperparâmetros para otimizar o `RandomForestClassifier`.

## ✨ Funcionalidades

- **Pipeline de Pré-processamento:** Uma função centralizada `preprocess` trata valores ausentes, converte dados categóricos e limpa colunas.
- **Engenharia de Features (Feature Engineering):** Criação de mais de 10 novas features, incluindo:
  - `FamilySize` e `IsAlone` (baseado em `SibSp` e `Parch`)
  - `Title` (extraído do nome)
  - `Deck` (extraído da cabine)
  - `IsChild` e `IsMother` (baseado em idade, sexo e parentes)
  - `FarePerPerson` (tarifa dividida pelo tamanho da família)
  - `ClassFareInteraction` (interação Pclass * Fare)
  - `AgeBin` e `FareBin` (discretização de idade e tarifa)
- **Busca de Hiperparâmetros:** Testa automaticamente diferentes combinações de `n_estimators` e `max_depth` para encontrar o `RandomForestClassifier` com melhor acurácia em um conjunto de validação.
- **Treinamento Otimizado:** Treina o modelo final com todos os dados de treino usando os melhores parâmetros encontrados na etapa de validação.
- **Geração de Submissão:** Produz o arquivo `submission.csv` no formato exigido pelo Kaggle.

## 🎲 Metodologia do Pipeline de ML

1.  **Carregamento:** Os dados de `train.csv` e `test.csv` são carregados em DataFrames `pandas`.
2.  **Análise Exploratória:** Uma verificação rápida da taxa de sobrevivência por sexo é impressa no console.
3.  **Pré-processamento e Feature Engineering:** A função `preprocess` é aplicada a ambos os conjuntos de dados. Ela executa toda a limpeza (ex: `fillna` para `Age` e `Fare`) e a criação de novas features.
4.  **Alinhamento de Colunas:** O script garante que o conjunto de teste tenha exatamente as mesmas colunas (dummies) que o conjunto de treino.
5.  **Validação (Grid Search Simplificado):** Os dados de treino são divididos (`train_test_split`) em treino e validação (80/20). Um loop `for` aninhado testa diferentes `n_estimators` e `max_depth`, treinando no subconjunto de treino e medindo a acurácia no de validação.
6.  **Seleção do Melhor Modelo:** A combinação de parâmetros com a maior acurácia de validação é armazenada.
7.  **Treinamento Final:** Um novo modelo `RandomForestClassifier` é instanciado com os melhores parâmetros e treinado usando **todos** os dados de treino (`X` e `y`).
8.  **Previsão:** O modelo final é usado para gerar previsões no conjunto de teste (`X_test`).
9.  **Geração do Arquivo:** As previsões são salvas no formato `PassengerId`, `Survived` no arquivo `submission.csv`.

## 🛠️ Tecnologias Utilizadas

- **Linguagem:** Python 3
- **Bibliotecas Principais:**
  - `pandas`: Para manipulação e análise dos dados.
  - `scikit-learn`: Para o modelo (`RandomForestClassifier`), divisão de dados (`train_test_split`) e métricas (`accuracy_score`).
  - `sys`: Para configurar o encoding de saída para UTF-8.

## ⚙️ Pré-requisitos

Antes de executar, você precisa ter o Python 3 instalado e as seguintes bibliotecas. Você pode instalá-las com pip:

```bash
pip install pandas scikit-learn
```

Além disso, você precisará dos arquivos de dados do Kaggle (`train.csv` e `test.csv`) em um subdiretório chamado `DesafioTitanic/`.

**Estrutura de arquivos esperada:**
```
├── DesafioTitanic/
│   ├── train.csv
│   ├── gender_submission.csv
│   ├── test.csv
│   └── main.py (este código)
└── README.md
```

## 🚀 Como Executar

1.  **Clone o repositório** ou salve o arquivo do projeto (`main.py`) em um diretório local.
2.  **Crie o diretório `DesafioTitanic/`** no mesmo nível do script.
3.  **Baixe os arquivos `train.csv` e `test.csv`** do Kaggle e coloque-os dentro de `DesafioTitanic/`.
4.  **Instale as dependências** conforme listado na seção de pré-requisitos.
5.  **Execute o script** principal via terminal:
    ```bash
    python main.py.py
    ```
6.  Aguarde a execução. O script imprimirá os resultados da busca de hiperparâmetros no console. Ao final, você verá a seguinte mensagem (se a última linha do script for descomentada):
    ```
    Arquivo 'submission.csv' criado com sucesso!
    ```

## 📂 Estrutura do Código

O script é organizado nas seguintes seções:

-   **Importação de bibliotecas:** Importa `pandas`, `sklearn` e `sys`.
-   **Configuração de Encoding:** Garante que a saída do console use UTF-8.
-   **Carregamento dos dados:** Lê os arquivos `train.csv` e `test.csv`.
-   **Análise Exploratória:** Impressão da taxa de sobrevivência por sexo.
-   **Função `preprocess`:** Definição da função principal de pré-processamento e engenharia de features.
-   **Aplicação do Pré-processamento:** Chamada da função `preprocess` para `train_data` e `test_data`.
-   **Alinhamento de Colunas:** Garante que os DataFrames de treino e teste tenham as mesmas colunas após o `get_dummies`.
-   **Definição de Features:** Seleciona a lista final de colunas a serem usadas no modelo.
-   **Separação para Validação:** Divide os dados de treino com `train_test_split`.
-   **Busca de Hiperparâmetros:** Loop `for` aninhado que testa `n_estimators` e `max_depth`.
-   **Treinamento do Modelo Final:** Treina o modelo com os melhores parâmetros em todos os dados de treino.
-   **Geração das Previsões:** Usa o modelo final para prever `test_data`.
-   **Criação do Arquivo de Submissão:** Salva os resultados no DataFrame `output`.

## 📄 Saídas Geradas

Ao executar o script, os seguintes arquivos serão criados (assumindo que a última linha seja descomentada):

-   `submission.csv`: O arquivo final pronto para ser enviado ao Kaggle, contendo o `PassengerId` e a previsão `Survived` (0 ou 1).

Além disso, o console exibirá:

-   O `head()` dos DataFrames de treino e teste.
-   A porcentagem de sobrevivência de homens e mulheres.
-   A lista de features usadas no modelo.
-   A acurácia de cada combinação de hiperparâmetro testada.
-   A melhor combinação de parâmetros encontrada e sua acurácia.

## 👨‍💻 Autor

- **Kevin Thiago dos Santos** - *Estudante de Ciência da Computação*
