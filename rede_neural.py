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
encoder = OneHotEncoder(sparse_output=False)  # Alterado de 'sparse=False' para 'sparse_output=False'
y_classe_encoded = encoder.fit_transform(y_classe.values.reshape(-1, 1))

# Abrir o arquivo para salvar as métricas
with open('resultados_metricas_rede_neural.txt', 'w') as f:
    for i in range(50):
        # Dividir os dados para regressão (gravidade)
        X_train, X_test, y_gravidade_train, y_gravidade_test, y_classe_train, y_classe_test = train_test_split(
            X_scaled, y_gravidade, y_classe_encoded, test_size=0.2, random_state=i)
        
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
        accuracy = accuracy_score(np.argmax(y_classe_test, axis=1) + 1, y_classe_pred)
        report = classification_report(np.argmax(y_classe_test, axis=1) + 1, y_classe_pred, output_dict=True)
        precision = np.mean([report[str(cls)]['precision'] for cls in np.unique(y_classe)])
        recall = np.mean([report[str(cls)]['recall'] for cls in np.unique(y_classe)])
        
        end_time_classificacao = time.time()
        elapsed_time_classificacao = end_time_classificacao - start_time_classificacao
        
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

print("Resultados salvos em 'resultados_metricas_rede_neural.txt'.")