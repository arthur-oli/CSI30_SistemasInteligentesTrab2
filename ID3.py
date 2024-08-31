import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import time

# Carregar o dataset
dataset = pd.read_csv('treino_sinais_vitais_com_label.txt', header=None)
dataset.columns = ['id', 'pressao_sistolica', 'pressao_diastolica', 'qualidade_pressao', 'pulso', 'respiracao', 'gravidade', 'classe']

# Separar as features e a variável alvo
X = dataset[['qualidade_pressao', 'pulso', 'respiracao', 'gravidade']]
y_classe = dataset['classe']

# Abrir o arquivo para salvar as métricas
with open('resultados_metricas_id3.txt', 'w') as f:
    for i in range(50):
        start_time = time.time()  # Iniciar cronômetro
        # Dividir o dataset em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y_classe, test_size=0.2, random_state=i)

        # Criar e treinar o modelo ID3 (DecisionTreeClassifier)
        id3_classifier = DecisionTreeClassifier(criterion='entropy', random_state=i)
        id3_classifier.fit(X_train, y_train)

        # Fazer previsões no conjunto de teste
        y_pred = id3_classifier.predict(X_test)

        # Calcular métricas de classificação
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        precision = np.mean([report[str(cls)]['precision'] for cls in np.unique(y_classe)])
        recall = np.mean([report[str(cls)]['recall'] for cls in np.unique(y_classe)])

        end_time = time.time()  # Parar cronômetro
        elapsed_time = end_time - start_time

        # Escrever as métricas no arquivo
        f.write(f"Execução {i+1}:\n")
        f.write(f"Tempo de Execução: {elapsed_time:.4f} segundos\n")
        f.write(f"Métricas de Classificação (Classe):\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Precision Média: {precision}\n")
        f.write(f"Recall Médio: {recall}\n")
        f.write("-" * 40 + "\n")

print("Resultados salvos em 'resultados_metricas_id3.txt'.")