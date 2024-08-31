import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, mean_absolute_error

# Carregar o dataset
dataset = pd.read_csv('treino_sinais_vitais_com_label.txt', header=None)
dataset.columns = ['id', 'pressao_sistolica', 'pressao_diastolica', 'qualidade_pressao', 'pulso', 'respiracao', 'gravidade', 'classe']

# Separar as features para gravidade e classe
X = dataset[['qualidade_pressao', 'pulso', 'respiracao']]
y_gravidade = dataset['gravidade']
y_classe = dataset['classe']


with open('resultados_metricas.txt', 'w') as f:
    for i in range(50):
        # Dividir o dataset em treino e teste para gravidade
        X_train, X_test, y_gravidade_train, y_gravidade_test = train_test_split(X, y_gravidade, test_size=0.2, random_state=i)

        # Criar e treinar o modelo Random Forest para prever a gravidade
        rf_regressor = RandomForestRegressor(n_estimators=100, random_state=i)
        rf_regressor.fit(X_train, y_gravidade_train)

        # Fazer previsões de gravidade no conjunto de teste
        y_gravidade_pred = rf_regressor.predict(X_test)

        # Calcular métricas de regressão
        mse = mean_squared_error(y_gravidade_test, y_gravidade_pred)
        mae = mean_absolute_error(y_gravidade_test, y_gravidade_pred)
        r2 = r2_score(y_gravidade_test, y_gravidade_pred)

        # Adicionar a gravidade prevista ao conjunto de treino e teste para classe
        X_train_classe = X_train.copy()
        X_train_classe['Gravidade_Prevista'] = rf_regressor.predict(X_train)

        X_test_classe = X_test.copy()
        X_test_classe['Gravidade_Prevista'] = y_gravidade_pred

        # Treinar o modelo Random Forest para prever a classe com a nova feature
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=i)
        rf_classifier.fit(X_train_classe, y_classe.loc[X_train.index])

        # Fazer previsões de classe no conjunto de teste
        y_classe_pred = rf_classifier.predict(X_test_classe)

        # Calcular métricas de classificação
        accuracy = accuracy_score(y_classe.loc[X_test.index], y_classe_pred)
        report = classification_report(y_classe.loc[X_test.index], y_classe_pred, output_dict=True)
        precision = np.mean([report[str(cls)]['precision'] for cls in np.unique(y_classe)])
        recall = np.mean([report[str(cls)]['recall'] for cls in np.unique(y_classe)])

        # Escrever as métricas no arquivo
        f.write(f"Execução {i+1}:\n")
        f.write(f"Métricas de Regressão (Gravidade):\n")
        f.write(f"MSE: {mse}\n")
        f.write(f"MAE: {mae}\n")
        f.write(f"R²: {r2}\n")
        f.write(f"Métricas de Classificação (Classe):\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Precision Média: {precision}\n")
        f.write(f"Recall Médio: {recall}\n")
        f.write("-" * 40 + "\n")

print("Resultados salvos em 'resultados_metricas.txt'.")