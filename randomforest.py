import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Carregar o dataset
dataset = pd.read_csv('treino_sinais_vitais_sem_label.txt', header=None)
dataset.columns = ['id', 'pressao_sistolica', 'pressao_diastolica', 'qualidade_pressao', 'pulso', 'respiracao', 'gravidade']

# Separar as features e a variável alvo
X = dataset[['qualidade_pressao', 'pulso', 'respiracao']]
y = dataset['gravidade']

# Dividir o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.12)

# Criar e treinar o modelo Random Forest
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# Avaliar o modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R² Score: {r2}')

# Criar um DataFrame com os resultados
resultados = pd.DataFrame({
    'Qualidade_Pressao': X_test['qualidade_pressao'],
    'Pulso': X_test['pulso'],
    'Respiracao': X_test['respiracao'],
    'Gravidade_Real': y_test,
    'Gravidade_Prevista': y_pred
})

# Salvar os resultados em um arquivo
resultados.to_csv('resultados_random_forest.txt', index=False, header=False)
