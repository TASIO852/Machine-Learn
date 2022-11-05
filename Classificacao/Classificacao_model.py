from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

arquivo = pd.read_csv(
    'C:/Users/Pcyes_User/OneDrive/Documentos/My projects/ML/Exel/wine_dataset.csv')

arquivo.head()

arquivo['style'] = arquivo['style'].replace('red', 0)
arquivo['style'] = arquivo['style'].replace('white', 1)

y = arquivo['style']
x = arquivo.drop('style', axis=1)

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3)

# Oque a Ia vai fazer eo modelo
modelo = ExtraTreesClassifier()
# Ia treinada (Pode ser uma lista aqui ou um banco de dados)
modelo.fit(x_treino, y_treino)

resultado = modelo.predict([x_teste], [y_teste])

# presisao da Ia (score)
resultado = modelo.score(x_teste, y_teste)
print(resultado)
