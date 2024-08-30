import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report

# Carregar o dataset
dataset = pd.read_csv('treino_sinais_vitais_com_label.txt', header=None)
dataset.columns = ['id', 'pressao_sistolica', 'pressao_diastolica', 'qualidade_pressao', 'pulso', 'respiracao', 'gravidade', 'classe']

# Separar as features para gravidade e classe
X = dataset[['qualidade_pressao', 'pulso', 'respiracao']]
y_gravidade = dataset['gravidade']
y_classe = dataset['classe']

# Dividir o dataset em treino e teste para gravidade
X_train, X_test, y_gravidade_train, y_gravidade_test = train_test_split(X, y_gravidade, test_size=0.2)

# Criar e treinar o modelo Random Forest para prever a gravidade
rf_regressor = RandomForestRegressor(n_estimators=100)
rf_regressor.fit(X_train, y_gravidade_train)

# Fazer previsões de gravidade no conjunto de teste
y_gravidade_pred = rf_regressor.predict(X_test)

# Adicionar a gravidade prevista ao conjunto de treino e teste para classe
X_train_classe = X_train.copy()
X_train_classe['Gravidade_Prevista'] = rf_regressor.predict(X_train)

X_test_classe = X_test.copy()
X_test_classe['Gravidade_Prevista'] = y_gravidade_pred

# Treinar o modelo Random Forest para prever a classe com a nova feature
rf_classifier = RandomForestClassifier(n_estimators=100)
rf_classifier.fit(X_train_classe, y_classe.loc[X_train.index])

# Fazer previsões de classe no conjunto de teste
y_classe_pred = rf_classifier.predict(X_test_classe)

# Avaliar o modelo de classificação
accuracy = accuracy_score(y_classe.loc[X_test.index], y_classe_pred)
report = classification_report(y_classe.loc[X_test.index], y_classe_pred)

print(f'Accuracy (Classe): {accuracy}')
print('Classification Report:')
print(report)

# Adicionar o ID original e as previsões de classe ao DataFrame e salvar os resultados
X_test_classe['ID_Original'] = dataset['id'].loc[X_test.index]
X_test_classe['Classe_Prevista'] = y_classe_pred
X_test_classe = X_test_classe[['ID_Original', 'qualidade_pressao', 'pulso', 'respiracao', 'Gravidade_Prevista', 'Classe_Prevista']]
X_test_classe = X_test_classe.sort_values(by='ID_Original')
X_test_classe.to_csv('resultados_random_forest.txt', index=False, header=False)