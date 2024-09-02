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

# Listas para armazenar as métricas
accuracy_list = []
precision_list = []
recall_list = []
time_list = []

# Abrir o arquivo para salvar as métricas
with open('resultados_metricas_id3.txt', 'w') as f:
    for i in range(50):
        start_time = time.time()  # Iniciar cronômetro
        # Dividir o dataset em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y_classe, test_size=0.3, random_state=i)

        # Criar e treinar o modelo ID3 (DecisionTreeClassifier)
        id3_classifier = DecisionTreeClassifier(criterion='entropy', random_state=i)
        id3_classifier.fit(X_train, y_train)

        # Fazer previsões no conjunto de teste
        y_pred = id3_classifier.predict(X_test)

        # Calcular métricas de classificação
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Filtrar as classes presentes no conjunto de teste
        present_classes = [str(cls) for cls in np.unique(y_test)]
        precision = np.mean([report[cls]['precision'] for cls in present_classes if cls in report])
        recall = np.mean([report[cls]['recall'] for cls in present_classes if cls in report])

        end_time = time.time()  # Parar cronômetro
        elapsed_time = end_time - start_time

        # Armazenar as métricas na lista
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        time_list.append(elapsed_time)

        # Escrever as métricas no arquivo
        f.write(f"Execução {i+1}:\n")
        f.write(f"Tempo de Execução: {elapsed_time:.4f} segundos\n")
        f.write(f"Métricas de Classificação (Classe):\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Precision Média: {precision}\n")
        f.write(f"Recall Médio: {recall}\n")
        f.write("-" * 40 + "\n")

# Calcular as médias das métricas
accuracy_mean = np.mean(accuracy_list)
precision_mean = np.mean(precision_list)
recall_mean = np.mean(recall_list)
time_mean = np.mean(time_list)

# Escrever as médias no início do arquivo
with open('resultados_metricas_id3.txt', 'r+') as f:
    content = f.read()
    f.seek(0, 0)
    f.write("MÉDIAS DAS MÉTRICAS APÓS 50 ITERAÇÕES:\n")
    f.write(f"Tempo Médio de Execução: {time_mean:.4f} segundos\n")
    f.write(f"Acurácia Média: {accuracy_mean}\n")
    f.write(f"Precisão Média: {precision_mean}\n")
    f.write(f"Recall Médio: {recall_mean}\n")
    f.write("=" * 40 + "\n\n")
    f.write(content)

print("Resultados salvos em 'resultados_metricas_id3.txt'.")
