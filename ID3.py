import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Carregar o dataset
dataset = pd.read_csv('treino_sinais_vitais_com_label.txt', header=None)
dataset.columns = ['id', 'pressao_sistolica', 'pressao_diastolica', 'qualidade_pressao', 'pulso', 'respiracao', 'gravidade', 'classe']

# Separar as features e a variável alvo
X = dataset[['qualidade_pressao', 'pulso', 'respiracao', 'gravidade']]
y = dataset['classe']

# Dividir o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar e treinar o modelo ID3
model = DecisionTreeClassifier(criterion='entropy', random_state=42)
model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# Avaliar o modelo
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)

# Criar um DataFrame com os resultados
resultados = pd.DataFrame({
    'Qualidade_Pressao': X_test['qualidade_pressao'],
    'Pulso': X_test['pulso'],
    'Respiracao': X_test['respiracao'],
    'Classe_Real': y_test,
    'Classe_Prevista': y_pred
})

# Salvar os resultados em um arquivo
resultados.to_csv('resultados_ID3.txt', index=False, header=False)
