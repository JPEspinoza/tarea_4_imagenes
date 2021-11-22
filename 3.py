"""
Este script hace el paso 3 de la tarea:
3. De la base de datos generada, utilice la técnica Take-L Plus-R sobre el 
conjunto de características maximizando el índice de Fisher

en la sopa de letras se ven las siguientes letras
a,b,c,e,f,g

asumimos 6 clusters para fisher de 5 observaciones cada uno
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

data = pd.read_csv("2.csv")

# visualizar datos con PCA para ver clusters
pca = PCA(2)
pca_data = pca.fit_transform(data.drop('cluster', axis=1))
print(pca_data)
plt.scatter(pca_data[:,0], pca_data[:,1], c=data[["cluster"]].to_numpy())
plt.show()