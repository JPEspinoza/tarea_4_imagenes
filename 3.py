"""
Este script hace el paso 3 de la tarea:
3. De la base de datos generada, utilice la técnica Take-L Plus-R sobre el 
conjunto de características maximizando el índice de Fisher


"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

# leemos el csv
# pandas no sabe leer numeros complejos, asi que los convertimos manualmente
data = pd.read_csv("2.csv")

# visualizar los datos con PCA
pca = PCA(3)
data = pca.fit_transform(data)

ax = plt.axes(projection="3d")
ax.scatter(data[:,0], data[:,1], data[:,2])
plt.show()



print(data)