"""
le os datasets
faz a DE com todos os pontos do training e o ponto do testing
define os k vizinhos com isso
faz a votação pra ver qual classe alvo aparece mais
repete pra todos os pontos do dataset de testes
"""

import math
import pandas as pd

def euclidean_distance(point1, point2):
    distance = 0.0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2
    return math.sqrt(distance)

def get_k_neighbors(training_data, testing_instance, k):
    distances = []
    for index, train_instance in training_data.iterrows():
        distance = euclidean_distance(train_instance.iloc[1:-1].values, testing_instance.iloc[1:-1].values) 
        distances.append((train_instance['ID'], distance, train_instance['class'])) 
    #print(distances)
    distances.sort(key=lambda x: x[1])  

    neighbors = distances[:k]
    print("\nK-Vizinhos:\n")
    print(neighbors)
    return neighbors

def voting(neighbors):
    pos = 0
    neg = 0
    for neighbor in neighbors:
        if neighbor[2] == 1:
            pos += 1
        else:
            neg += 1
    return 1 if pos > neg else 0

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
t = 4
k = 1

training_data = pd.read_csv(file_path[t-1], sep='\t')
testing_data = pd.read_csv(file_path[t+3], sep='\t')
print(training_data.head(45))
print(testing_data.head())


for i in range(4):
    acerto = 0 
    total = 0
    print(f"Valor de K = {k}")
    for index, testing_instance in testing_data.iterrows():
        neighbors = get_k_neighbors(training_data, testing_instance, k)
        elected_class = voting(neighbors)
        print(f"Teste {index + 1}: Classe eleita = {elected_class}, Classe real: = {testing_instance['class']}")
        if elected_class == testing_instance['class']:
            acerto += 1
        total += 1

    acuracia = acerto / total
    print(f"Acuracia: {acuracia}")
    k +=2


"""
a)
AT01_KNN/datasets/Dados_Originais_2Features/TrainingData_2F_Original.txt
Valor de K = 1
Teste 1: Classe eleita = 0, Classe real: = 0
Teste 2: Classe eleita = 0, Classe real: = 0
Teste 3: Classe eleita = 1, Classe real: = 1
Teste 4: Classe eleita = 0, Classe real: = 1
Acuracia: 0.75
Valor de K = 3
Teste 1: Classe eleita = 1, Classe real: = 0
Teste 2: Classe eleita = 0, Classe real: = 0
Teste 3: Classe eleita = 1, Classe real: = 1
Teste 4: Classe eleita = 0, Classe real: = 1
Acuracia: 0.5
Valor de K = 5
Teste 1: Classe eleita = 1, Classe real: = 0
Teste 2: Classe eleita = 1, Classe real: = 0
Teste 3: Classe eleita = 1, Classe real: = 1
Teste 4: Classe eleita = 0, Classe real: = 1
Acuracia: 0.25
Valor de K = 7
Teste 1: Classe eleita = 1, Classe real: = 0
Teste 2: Classe eleita = 1, Classe real: = 0
Teste 3: Classe eleita = 1, Classe real: = 1
Teste 4: Classe eleita = 0, Classe real: = 1
Acuracia: 0.25

AT01_KNN/datasets/Dados_Normalizados_2Features/TrainingData_2F_Norm.txt
Valor de K = 1
Teste 1: Classe eleita = 1, Classe real: = 0
Teste 2: Classe eleita = 1, Classe real: = 0
Teste 3: Classe eleita = 1, Classe real: = 1
Teste 4: Classe eleita = 1, Classe real: = 1
Acuracia: 0.5
Valor de K = 3
Teste 1: Classe eleita = 1, Classe real: = 0
Teste 2: Classe eleita = 1, Classe real: = 0
Teste 3: Classe eleita = 1, Classe real: = 1
Teste 4: Classe eleita = 0, Classe real: = 1
Acuracia: 0.25
Valor de K = 5
Teste 1: Classe eleita = 0, Classe real: = 0
Teste 2: Classe eleita = 1, Classe real: = 0
Teste 3: Classe eleita = 1, Classe real: = 1
Teste 4: Classe eleita = 1, Classe real: = 1
Acuracia: 0.75
Valor de K = 7
Teste 1: Classe eleita = 0, Classe real: = 0
Teste 2: Classe eleita = 0, Classe real: = 0
Teste 3: Classe eleita = 1, Classe real: = 1
Teste 4: Classe eleita = 1, Classe real: = 1
Acuracia: 1.0

    or      norm
1   0.75    0.5
3   0.5     0.25
5   0.25    0.75
7   0.25    1.0

b)
Valor de K = 5
N1 -> x = 26 y = 0
K-Vizinhos:
[('T21', 0.4, 0),  -> x = 26 y = 0.4
('T27', 1.0000499987500624, 1), -> x = 25 y = 0.01
('T30', 1.113597772986279, 1), -> x = 25 y = 0.49
('T25', 2.0143485299222674, 1), -> x = 28 y = 0.24
('T11', 4.000049999687504, 0)] -> x = 30 y = 0.02
Classe eleita = 1, Classe real: = 0

Analise: os valores de X foram responsáveis por determinar a distância.
y acabou fcando quase que irrelevante para o calculo por ser muito menor.

N2
K-Vizinhos:
[('T10', 1.0594810050208545, 0), 
('T37', 2.0012246250733576, 1), 
('T4', 5.508175741568165, 0), 
('T36', 6.003007579538776, 1), 
('T31', 9.000138887817231, 1)]    
Classe eleita = 1, Classe real: = 0

N3
K-Vizinhos:
[('T29', 0.010000000000000009, 1), 
('T28', 1.000199980003999, 1), 
('T15', 2.0567936211491906, 0), 
('T41', 3.0004166377354995, 1), 
('T35', 4.000799920015996, 1)]
Classe eleita = 1, Classe real: = 1

N4
K-Vizinhos:
[('T9', 1.0846197490365, 0), 
('T26', 2.010994778710278, 1), 
('T16', 2.08806130178211, 0), 
('T11', 2.106086417980041, 0), 
('T25', 4.024127234569007, 1)]
Classe eleita = 0, Classe real: = 1

c)
AT01_KNN/datasets/Dados_Normalizados_2Features/TrainingData_2F_Norm.txt
citric acid = 1.0
[('T32', 0.08700000000000002, 1), 
('T7', 0.14200352108310554, 0), 
('T8', 0.16131955864060624, 0), 
('T38', 0.25770137756713674, 1), 
('T30', 0.28436947796836426, 1)]
Classe eleita = 1, Classe real: = 1

citric acid = 0.3
[('T25', 0.061400325732035, 1), 
('T9', 0.08238931969618393, 0), 
('T19', 0.10104454463255301, 0), 
('T20', 0.10176934705499491, 0), 
('T39', 0.11637869220780925, 1)]
Teste 4: Classe eleita = 0, Classe real: = 1

citric acid = 0.85
[('T7', 0.08523496934944015, 0), 
('T8', 0.1145600279329575, 0), 
('T30', 0.14023551618616448, 1), 
('T44', 0.14356183336806477, 1), 
('T14', 0.15559562975867927, 0)]
Teste 4: Classe eleita = 0, Classe real: = 1


AT01_KNN/datasets/Dados_Normalizados_11Features/TrainingData_11F_Norm.txt
citric acid = 1.0
[('T30', 1.9241842427376852, 1), 
('T1', 2.045333713602746, 0), 
('T8', 2.04690546923887, 0), 
('T7', 2.462202266264898, 0), 
('T6', 2.4982051557067924, 0)]
Classe eleita = 0, Classe real: = 1

citric acid = 0.3
[('T30', 1.949842301315673, 1), 
('T1', 2.0139488573446944, 0), 
('T8', 2.124763986893603, 0), 
('T6', 2.472575378021871, 0), 
('T7', 2.5272989534283434, 0)]       
Teste 4: Classe eleita = 0, Classe real: = 1

citric acid = 0.85
[('T30', 1.9082151346218799, 1), 
('T1', 2.0183136525327274, 0), 
('T8', 2.043751941895102, 0), 
('T7', 2.45958126517503, 0), 
('T6', 2.476131862401516, 0)]        
Teste 4: Classe eleita = 0, Classe real: = 1

"""