# Pedi um código pronto de KNN para o chat GPT para testar e comparar os resultados com o código desenvolvido por mim
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from collections import Counter

# Função para prever a classe com base nos k vizinhos mais próximos
def knn_predict(training_data, test_data, k):
    predictions = []
    
    # Separa as colunas de características e classes
    X_train = training_data.iloc[:, 1:-1].values  # Todas as colunas exceto ID e classe
    y_train = training_data.iloc[:, -1].values     # Última coluna (classe)
    X_test = test_data.iloc[:, 1:-1].values        # Todas as colunas exceto ID e classe
    
    # Calcula as distâncias entre os pontos de teste e de treino
    distances = pairwise_distances(X_test, X_train, metric='euclidean')
    
    # Para cada ponto de teste
    for dist in distances:
        # Obtém os k menores índices (vizinhos mais próximos)
        k_nearest_indices = np.argsort(dist)[:k]
        # Obtém as classes correspondentes dos k vizinhos mais próximos
        k_nearest_classes = y_train[k_nearest_indices]
        # Prediz a classe mais comum (maioria)
        most_common = Counter(k_nearest_classes).most_common(1)[0][0]
        predictions.append(most_common)
    
    return np.array(predictions)

# Carregando os dados de treino e teste usando pandas
training_data = pd.read_csv('AT01_KNN/datasets/Dados_Normalizados_11Features/TrainingData_11F_Norm.txt', sep='\t')
testing_data = pd.read_csv('AT01_KNN/datasets/Dados_Normalizados_11Features/TestingData_11F_Norm.txt', sep='\t')

# Testando com k = 1, 3, 5, 7
for k in [1, 3, 5, 7]:
    predicted_classes = knn_predict(training_data, testing_data, k)
    print(f'Previsões com k={k}: {predicted_classes}')
