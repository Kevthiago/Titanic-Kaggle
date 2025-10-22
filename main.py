# ========================================
# Importando bibliotecas necessárias
# ========================================
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys

# Configurando a saída para UTF-8 (para evitar erros de caractere)
sys.stdout.reconfigure(encoding='utf-8')

# ========================================
# Carregando os dados
# ========================================
train_data = pd.read_csv('DesafioTitanic/train.csv')
test_data = pd.read_csv('DesafioTitanic/test.csv')

print("Dados de Treino:")
print(train_data.head())
print("\nDados de Teste:")
print(test_data.head())

# ========================================
# Explorando um padrão - Sexo
# ========================================
women = train_data.loc[train_data.Sex == 'female']["Survived"]
men = train_data.loc[train_data.Sex == 'male']["Survived"]

print(f"\nPorcentagem (%) de mulheres que sobreviveram: {sum(women)/len(women):.2f}")
print(f"Porcentagem (%) de homens que sobreviveram: {sum(men)/len(men):.2f}")

# ========================================
# Pré-processamento (aplicado em ambos os conjuntos)
# ========================================

def preprocess(df):
    # Conversões básicas
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Ticket'] = df['Ticket'].astype('category').cat.codes

    # Valores ausentes
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna('S')

    # =========================
    # FEATURE ENGINEERING
    # =========================

    # --- Tamanho da família e solidão ---
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # --- Título extraído do nome ---
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    title_mapping = {
        "Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Dr": 4, "Rev": 5,
        "Col": 6, "Major": 7, "Mlle": 1, "Ms": 1, "Mme": 2,
        "Countess": 9, "Lady": 9, "Jonkheer": 10, "Don": 10, 
        "Dona": 10, "Capt": 10
    }
    df['Title'] = df['Title'].map(title_mapping).fillna(11)

    # --- Deck (primeira letra da cabine) ---
    df['Deck'] = df['Cabin'].fillna('U').astype(str).str[0]
    df['Deck'] = df['Deck'].astype('category').cat.codes

    # --- Criança ---
    df['IsChild'] = (df['Age'] < 12).astype(int)

    # --- Mãe ---
    df['IsMother'] = 0
    df.loc[(df['Sex'] == 1) & (df['Parch'] > 0) & (df['Age'] > 18) & (df['Title'] == 2), 'IsMother'] = 1

    # --- Tarifa por pessoa ---
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']

    # --- Interação entre classe e tarifa ---
    df['ClassFareInteraction'] = df['Pclass'] * df['Fare']

    # --- Faixas discretas de idade e tarifa ---
    df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 20, 40, 60, 80], labels=False)
    df['FareBin'] = pd.qcut(df['Fare'], 4, labels=False)

    # --- One-hot encoding de Embarked ---
    df = pd.get_dummies(df, columns=['Embarked'], prefix='Emb')

    # --- Limpeza de colunas redundantes ---
    df = df.drop(columns=['Name', 'Cabin', 'Ticket'], errors='ignore')

    return df

# Aplicando o pré-processamento
train_data = preprocess(train_data)
test_data = preprocess(test_data)

# ========================================
# Garantindo que as colunas sejam as mesmas nos dois conjuntos
# ========================================
missing_cols = set(train_data.columns) - set(test_data.columns)
for col in missing_cols:
    test_data[col] = 0
test_data = test_data.reindex(columns=train_data.columns.drop('Survived'), fill_value=0)

# ========================================
# Definição das features e variável alvo
# ========================================
features = [
    'Pclass', 'Sex', 'Age', 'Fare',
    'FamilySize', 'IsAlone', 'IsChild', 'IsMother',
    'Title', 'Deck',
    'FarePerPerson', 'ClassFareInteraction',
    'AgeBin', 'FareBin'
]
features += [col for col in train_data.columns if col.startswith('Emb_')]

print("\nFeatures usadas no modelo:")
print(features)

X = train_data[features]
y = train_data["Survived"]

# ========================================
# Teste de acurácia local (validação)
# ========================================
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ========================================
# Teste de diferentes combinações de RandomForest
# ========================================
n_estimators_list = [100, 160, 200, 250]  # diferentes tamanhos da floresta
max_depth_list = [6, 8, 10, 12]           # diferentes profundidades máximas

best_score = 0
best_params = {}

for n in n_estimators_list:
    for d in max_depth_list:
        # Instanciando o modelo
        model = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=1)
        
        # Treino e validação
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        score = accuracy_score(y_val, y_pred)
        
        print(f"n_estimators={n}, max_depth={d}, acurácia={score:.4f}")
        
        # Guardando o melhor
        if score > best_score:
            best_score = score
            best_params = {'n_estimators': n, 'max_depth': d}

print("\nMelhor combinação encontrada:")
print(f"n_estimators={best_params['n_estimators']}, max_depth={best_params['max_depth']}, acurácia={best_score:.4f}")

# ========================================
# Treinando modelo final com todos os dados
# ========================================
model = RandomForestClassifier(
    n_estimators=best_params['n_estimators'], 
    max_depth=best_params['max_depth'], 
    random_state=1
)
model.fit(X, y)  # usa TODOS os dados de treino

# ========================================
# Gerando previsões para o dataset de teste
# ========================================
X_test = test_data[features]
predictions = model.predict(X_test)
print("\nPrevisões realizadas com sucesso!")

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
#output.to_csv('DesafioTitanic/submission.csv', index=False)
print("\nArquivo 'submission.csv' criado com sucesso!")