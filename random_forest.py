import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, mean_absolute_error
import time

# Carregar o dataset
dataset = pd.read_csv('treino_sinais_vitais_com_label.txt', header=None)
dataset.columns = ['id', 'pressao_sistolica', 'pressao_diastolica', 'qualidade_pressao', 'pulso', 'respiracao', 'gravidade', 'classe']

# Separar as features para gravidade e classe
X = dataset[['qualidade_pressao', 'pulso', 'respiracao']]
y_gravidade = dataset['gravidade']
y_classe = dataset['classe']

# Listas para armazenar as métricas
mse_list = []
mae_list = []
r2_list = []
accuracy_list = []
precision_list = []
recall_list = []
time_regressao_list = []
time_classificacao_list = []

with open('resultados_metricas_random_forest.txt', 'w') as f:
    for i in range(50):
        print("Iniciando iteração ", i)
        start_time_regressao = time.time()
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

        end_time_regressao = time.time()
        elapsed_time_regressao = end_time_regressao - start_time_regressao

        # Adicionar a gravidade prevista ao conjunto de treino e teste para classe
        X_train_classe = X_train.copy()
        X_train_classe['Gravidade_Prevista'] = rf_regressor.predict(X_train)

        X_test_classe = X_test.copy()
        X_test_classe['Gravidade_Prevista'] = y_gravidade_pred

        start_time_classificacao = time.time()

        # Treinar o modelo Random Forest para prever a classe com a nova feature
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=i)
        rf_classifier.fit(X_train_classe, y_classe.loc[X_train.index])

        # Fazer previsões de classe no conjunto de teste
        y_classe_pred = rf_classifier.predict(X_test_classe)

        # Calcular métricas de classificação
        accuracy = accuracy_score(y_classe.loc[X_test.index], y_classe_pred)
        report = classification_report(y_classe.loc[X_test.index], y_classe_pred, output_dict=True)
        
        # Filtrar classes presentes no relatório
        present_classes = [str(cls) for cls in np.unique(y_classe.loc[X_test.index])]
        precision = np.mean([report[cls]['precision'] for cls in present_classes if cls in report])
        recall = np.mean([report[cls]['recall'] for cls in present_classes if cls in report])

        end_time_classificacao = time.time()
        elapsed_time_classificacao = end_time_classificacao - start_time_classificacao

        # Armazenar as métricas na lista
        mse_list.append(mse)
        mae_list.append(mae)
        r2_list.append(r2)
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        time_regressao_list.append(elapsed_time_regressao)
        time_classificacao_list.append(elapsed_time_classificacao)

        # Escrever as métricas no arquivo
        f.write(f"Execução {i+1}:\n")
        f.write(f"Tempo de Execução (Regressão): {elapsed_time_regressao:.4f} segundos\n")
        f.write(f"Tempo de Execução (Classificação): {elapsed_time_classificacao:.4f} segundos\n")
        f.write(f"Métricas de Regressão (Gravidade):\n")
        f.write(f"MSE: {mse}\n")
        f.write(f"MAE: {mae}\n")
        f.write(f"R²: {r2}\n")
        f.write(f"Métricas de Classificação (Classe):\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Precision Média: {precision}\n")
        f.write(f"Recall Médio: {recall}\n")
        f.write("-" * 40 + "\n")

# Calcular as médias das métricas
mse_mean = np.mean(mse_list)
mae_mean = np.mean(mae_list)
r2_mean = np.mean(r2_list)
accuracy_mean = np.mean(accuracy_list)
precision_mean = np.mean(precision_list)
recall_mean = np.mean(recall_list)
time_regressao_mean = np.mean(time_regressao_list)
time_classificacao_mean = np.mean(time_classificacao_list)

# Escrever as médias no início do arquivo
with open('resultados_metricas_random_forest.txt', 'r+') as f:
    content = f.read()
    f.seek(0, 0)
    f.write("MÉDIAS DAS MÉTRICAS APÓS 50 ITERAÇÕES:\n")
    f.write(f"Tempo Médio de Execução (Regressão): {time_regressao_mean:.4f} segundos\n")
    f.write(f"Tempo Médio de Execução (Classificação): {time_classificacao_mean:.4f} segundos\n")
    f.write(f"MSE Médio: {mse_mean}\n")
    f.write(f"MAE Médio: {mae_mean}\n")
    f.write(f"R² Médio: {r2_mean}\n")
    f.write(f"Acurácia Média: {accuracy_mean}\n")
    f.write(f"Precisão Média: {precision_mean}\n")
    f.write(f"Recall Médio: {recall_mean}\n")
    f.write("=" * 40 + "\n\n")
    f.write(content)

print("Resultados salvos em 'resultados_metricas_random_forest.txt'.")
