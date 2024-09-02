import pandas as pd
import numpy as np
import time
from keras import models, layers, optimizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report

# Carregar o dataset
dataset = pd.read_csv('treino_sinais_vitais_com_label.txt', header=None)
dataset.columns = ['id', 'pressao_sistolica', 'pressao_diastolica', 'qualidade_pressao', 'pulso', 'respiracao', 'gravidade', 'classe']

# Separar as features e os alvos
X = dataset[['qualidade_pressao', 'pulso', 'respiracao']]
y_gravidade = dataset['gravidade']
y_classe = dataset['classe']

# Escalonamento das features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# One-hot encoding para a classe
encoder = OneHotEncoder(sparse_output=False)
y_classe_encoded = encoder.fit_transform(y_classe.values.reshape(-1, 1))

# Listas para armazenar as métricas
mse_list = []
mae_list = []
r2_list = []
accuracy_list = []
precision_list = []
recall_list = []
time_regressao_list = []
time_classificacao_list = []

# Abrir o arquivo para salvar as métricas
with open('resultados_metricas_rede_neural.txt', 'w') as f:
    for i in range(50):
        print("Iniciando iteração ", i)
        # Dividir os dados para regressão (gravidade)
        X_train, X_test, y_gravidade_train, y_gravidade_test, y_classe_train, y_classe_test = train_test_split(
            X_scaled, y_gravidade, y_classe_encoded, test_size=0.1, random_state=i)
        
        # Criar o modelo de rede neural para regressão
        model_reg = models.Sequential()
        model_reg.add(layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
        model_reg.add(layers.Dense(32, activation='relu'))
        model_reg.add(layers.Dense(1, activation='linear'))  # Saída para gravidade
        
        # Compilar o modelo de regressão
        model_reg.compile(optimizer=optimizers.Adam(), loss='mse')
        
        # Medir tempo de treinamento da regressão
        start_time_regressao = time.time()
        
        # Treinar o modelo de regressão
        model_reg.fit(X_train, y_gravidade_train, epochs=50, verbose=0)
        
        # Fazer previsões de gravidade
        y_gravidade_pred = model_reg.predict(X_test).flatten()
        
        # Calcular métricas de regressão
        mse = mean_squared_error(y_gravidade_test, y_gravidade_pred)
        mae = mean_absolute_error(y_gravidade_test, y_gravidade_pred)
        r2 = r2_score(y_gravidade_test, y_gravidade_pred)
        
        end_time_regressao = time.time()
        elapsed_time_regressao = end_time_regressao - start_time_regressao
        
        # Adicionar a gravidade prevista ao conjunto de treino e teste para classe
        X_train_classe = np.hstack([X_train, model_reg.predict(X_train).reshape(-1, 1)])
        X_test_classe = np.hstack([X_test, y_gravidade_pred.reshape(-1, 1)])
        
        # Criar o modelo de rede neural para classificação
        model_clf = models.Sequential()
        model_clf.add(layers.Dense(64, activation='relu', input_shape=(X_train_classe.shape[1],)))
        model_clf.add(layers.Dense(32, activation='relu'))
        model_clf.add(layers.Dense(4, activation='softmax'))  # 4 classes
        
        # Compilar o modelo de classificação
        model_clf.compile(optimizer=optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Medir tempo de treinamento da classificação
        start_time_classificacao = time.time()
        
        # Treinar o modelo de classificação
        model_clf.fit(X_train_classe, y_classe_train, epochs=50, verbose=0)
        
        # Fazer previsões de classe
        y_classe_pred = model_clf.predict(X_test_classe)
        y_classe_pred = np.argmax(y_classe_pred, axis=1) + 1  # Converter de one-hot para labels
        
        # Calcular métricas de classificação
        y_classe_test_labels = np.argmax(y_classe_test, axis=1) + 1  # Converter de one-hot para labels
        accuracy = accuracy_score(y_classe_test_labels, y_classe_pred)
        report = classification_report(y_classe_test_labels, y_classe_pred, output_dict=True)
        
        # Filtrar as classes presentes no conjunto de teste
        present_classes = [str(cls) for cls in np.unique(y_classe_test_labels)]
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
with open('resultados_metricas_rede_neural.txt', 'r+') as f:
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

print("Resultados salvos em 'resultados_metricas_rede_neural.txt'.")
