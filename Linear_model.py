from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
import pandas as pd
import numpy as np

porco1 = [0, 1, 0]  # Amostras
porco2 = [0, 1, 1]  # Amostras
porco3 = [1, 1, 0]  # Amostras

cachorro1 = [0, 1, 1]  # Amostras
cachorro2 = [1, 0, 1]  # Amostras
cachorro3 = [1, 1, 1]  # Amostras

# Pre-classificaçao das amostras para treino do modelo real
treino_x = [porco1, porco2, porco3, cachorro1, cachorro2, cachorro3]
treino_y = [1, 1, 1, 0, 0, 0] # 0 para cavhorro e 1 para porco

modelo = LinearSVC()  # Modelo a ser implementado
modelo.fit(treino_x, treino_y)

animal = [1, 1, 1]  # Dataset real a ser trabalhado
print(modelo.predict([animal])) # resultado da classificaçao

c1 = [0, 0, 1]
c2 = [0, 1, 0]
c3 = [0, 1, 1]

teste_x = [c1, c2, c3]
teste_y = [0, 1, 1]

preview = modelo.predict(teste_x)
print(preview)

# acuracia taxa de acerto
score = accuracy_score(teste_y, preview)

print(f'{score*100} % ')
