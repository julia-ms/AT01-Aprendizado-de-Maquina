"""
le os datasets
faz a DE com todos os pontos do traing e o ponto do test
define os k vizinhos com isso
faz a votação pra ver qual classe alvo aparece mais

"""

import math
import pandas as pd

def euclidean_distance(point1, point2):
    distance = 0.0
    for i in range(len(point1) - 1):  # -1 pra excluir o target
        distance += (point1[i] - point2[i]) ** 2
    return math.sqrt(distance)

def get_k_neighbors(training_data, testing_data, k):
    distances = []
    for instances in training_data:
        distance = euclidean_distance(training_data, testing_data)
        distances.append(distance)
    distances.sort(key=lambda x: x[1])  # chat gpt me ensinou essa linha
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][0]) # to salvando só os index dos k vizinhos
    return neighbors

def voting(training_data, neighbors):
    pos = 0
    neg = 0
    for index, (train_row, distance) in enumerate(neighbors):
        print(index + " " + train_row + " " + distance + "/n")


    

file_path = [
    "AT01_KNN/datasets/Dados_Originais_2Features/TrainingData_2F_Original.txt",
    "AT01_KNN/datasets/Dados_Normalizados_2Features/TrainingData_2F_Norm.txt",
    "AT01_KNN/datasets/Dados_Originais_11Features/TrainingData_11F_Original.txt",
    "AT01_KNN/datasets/Dados_Normalizados_11Features/TrainingData_11F_Norm.txt",
    "AT01_KNN/datasets/Dados_Originais_2Features/TestingData_2F_Original.txt",
    "AT01_KNN/datasets/Dados_Normalizados_2Features/TestingData_2F_Norm.txt",
    "AT01_KNN/datasets/Dados_Originais_11Features/TestingData_11F_Original.txt",
    "AT01_KNN/datasets/Dados_Normalizados_11Features/TestingData_11F_Norm.txt",
]

# só pra saber qual dos 4 testes é
t = 1
k = 3


training_data = pd.read_csv(file_path[t-1], sep='\t')
testing_data = pd.read_csv(file_path[t+3], sep='\t')


for index, testing_instance in testing_data.iterrows():
    neighbors = get_k_neighbors(training_data, testing_instance, k)
    elected_class = voting(neighbors)
    print(f"Teste {index}: Classe eleita = {elected_class}")
