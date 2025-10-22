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

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    title_mapping = {
        "Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Dr": 4, "Rev": 5,
        "Col": 6, "Major": 7, "Mlle": 1, "Ms": 1, "Mme": 2,
        "Countess": 9, "Lady": 9, "Jonkheer": 10, "Don": 10,
        "Dona": 10, "Capt": 10
    }
    df['Title'] = df['Title'].map(title_mapping).fillna(11)

    df['Deck'] = df['Cabin'].fillna('U').astype(str).str[0]
    df['Deck'] = df['Deck'].astype('category').cat.codes

    df['IsChild'] = (df['Age'] < 12).astype(int)
    df['IsMother'] = 0
    df.loc[(df['Sex'] == 1) & (df['Parch'] > 0) & (df['Age'] > 18) & (df['Title'] == 2), 'IsMother'] = 1

    df['FarePerPerson'] = df['Fare'] / df['FamilySize']
    df['ClassFareInteraction'] = df['Pclass'] * df['Fare']

    df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 20, 40, 60, 80], labels=False)
    df['FareBin'] = pd.qcut(df['Fare'], 4, labels=False)

    df = pd.get_dummies(df, columns=['Embarked'], prefix='Emb')
    df = df.drop(columns=['Name', 'Cabin', 'Ticket'], errors='ignore')

    return df

# Aplicando o pré-processamento
train_data = preprocess(train_data)
test_data = preprocess(test_data)

# ========================================
# Garantindo colunas iguais
# ========================================
missing_cols = set(train_data.columns) - set(test_data.columns)
for col in missing_cols:
    test_data[col] = 0
test_data = test_data.reindex(columns=train_data.columns.drop('Survived'), fill_value=0)

# ========================================
# Definição de features e alvo
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
# Validação local
# ========================================
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ========================================
# Teste de combinações de RandomForest
# ========================================
n_estimators_list = [120, 140, 160, 180, 200]
max_depth_list = [8, 9, 10, 11]

best_score = 0
best_params = {}

print("\n🔍 Testando combinações de parâmetros...")
for n in n_estimators_list:
    for d in max_depth_list:
        model = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        score = accuracy_score(y_val, y_pred)

        print(f"n_estimators={n:<3} | max_depth={d:<2} | acurácia={score:.4f}")

        if score > best_score:
            best_score = score
            best_params = {'n_estimators': n, 'max_depth': d}

print("\n🏆 Melhor combinação encontrada:")
print(f"n_estimators={best_params['n_estimators']}, max_depth={best_params['max_depth']}, acurácia={best_score:.4f}")

# ========================================
# Modelo final com todos os dados
# ========================================
model = RandomForestClassifier(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    random_state=1
)
model.fit(X, y)

# ========================================
# Previsões no conjunto de teste
# ========================================
X_test = test_data[features]
predictions = model.predict(X_test)
print("\n✅ Previsões realizadas com sucesso!")

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('DesafioTitanic/submission.csv', index=False)
print("\n💾 Arquivo 'submission.csv' criado com sucesso!")
