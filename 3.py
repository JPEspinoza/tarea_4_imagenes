"""
Este script hace el paso 3 de la tarea:
3. De la base de datos generada, utilice la técnica Take-L Plus-R sobre el 
conjunto de características maximizando el índice de Fisher
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from numpy.matlib import repmat

data = pd.read_csv("features_scaled.csv")

"""
# if you want to find out features for not scaled data
# we ignore the imaginary part for ease of use
data = pd.read_csv(
    "features.csv",
    converters={
        'fourier0': lambda s: np.real(complex(s)),
        'fourier1': lambda s: np.real(complex(s)),
        'fourier2': lambda s: np.real(complex(s)),
        'fourier3': lambda s: np.real(complex(s)),
        'fourier4': lambda s: np.real(complex(s)),
    }
)
"""

classes = pd.read_csv('clase_letras.csv').to_numpy().ravel()

# visualizar datos con PCA para ver clusters
# esto es SOLO para visualizar
pca = PCA(2)
pca_data = pca.fit_transform(data)
plt.figure()
plt.title("PCA")
plt.scatter(pca_data[:,0], pca_data[:,1], c=classes)

# calcular fisher
def fisher_extraction(data, classes):
    # función calcula el índice de Fisher para un determinado 
    # conjunto de datos.
    # Input: data: lista con submatrices (una por cada clase)
    #      clases: lista con valores de las clases.

    data = data.to_numpy()
    classes = np.hstack(classes)

    # unimos los datos en una sola matriz
    # Buscamos la media de todos los datos
    Vm = np.mean(data, axis =0)
    # numero de columnas
    cols = data.shape[1] 

    n_classes = len(set(classes))
    n_obs = len(data)

    # convertimos los datos en un tensor
    temp = []

    for i in range(n_classes):
        idx = (classes==i)
        temp.append(data[idx])
    data = np.array(temp)

    # quedan los datos en un tensor
    # primer indice indica la clase 
    # segundo la observacion
    # tercero la carasteristica
    # data[clase, obs, carasteristica]

    # inicializacion de matrices
    p = np.zeros((n_classes,1))
    Vk = np.zeros((n_classes,cols))
    Gn = []
    
    # Centrado
    for i in range(n_classes):
        Vk[i,:] = np.mean(data[i], axis=0)-Vm    # centramos las medias de cada clase
        pts = data[i].shape[0]                   # numero de puntos de ese clase
        Gn.append(data[i]-repmat(Vm,pts,1))      # centramos los puntos de cada clase
        p[i] = data[i].shape[0] / n_obs          # probabilidad de cada cluster

    # Inicialización
    Cb = np.zeros((cols,cols))
    Cw = np.zeros((cols,cols))

    # construccion de matrices inter e intraclase
    for k in range(n_classes):
        Cb = Cb + p[k]*np.matmul((Vk[k,:]-Vm).reshape(cols,1), (Vk[k,:]-Vm).reshape(1,cols))
        MGn = np.array(Gn[k])
        Cw = Cw + p[k]*np.cov(MGn.T)

    #Calculamos el índice de Fisher
    try:
        J = np.trace(np.matmul(np.linalg.inv(Cw),Cb))
        return J
    except:
        return 0

"""
# exhaustive search because we can
# this takes about 4 hours to run on an i7 9700
from multiprocessing import Pool
from itertools import combinations, chain
import psutil 

# lower priority to not hang the PC
parent = psutil.Process()
parent.nice(10)

# wrapper function to parallelize extracting fisher
def exhaustive_search(combination):
    combination = list(combination)
    score = fisher_extraction(data[combination], classes)

    return {"score": score, "combination": combination}

# get all combinations for all sizes
all_combinations = chain.from_iterable(
    combinations(data.columns, i) for i in range(len(data.columns)+1)
)

scores = {}

# get all scores in parallel
with Pool(processes=8) as pool:
    scores = pool.map(exhaustive_search, all_combinations)

# find the highest score

highest = {"score": 0}
for score in scores:
    if highest["score"] < score["score"]:
        highest = score

print(highest)
# result: the highest score uses ALL features
"""

# do plus-l take-r
# we go up to 5 columns
# then we go down to 2

# forward selection
features_selected = []
best_score = 0

while len(features_selected) < 23:

    best_feature = ""
    for new_feature in data.columns:
        new_features = features_selected + [new_feature]

        new_score = fisher_extraction(data[new_features], classes)

        if new_score > best_score:
            best_feature = new_feature
            best_score = new_score
        
    features_selected = features_selected + [best_feature]

    print(f"new best feature: {best_feature}, score: {best_score}")
    print(f"current features: {features_selected}\n")

print(f"SFS finished, current features: {features_selected}\n")

# backwards selection
while len(features_selected) > 23:

    # every cycle we check the score with a feature removed
    # we remove the feature where the other features had the best score
    best_score = 0
    worst_feature = ""

    for feature in features_selected:

        # do deep copy
        new_features = features_selected[:]
        new_features.remove(feature)

        new_score = fisher_extraction(data[new_features], classes)

        if new_score > best_score:

            worst_feature = feature
            best_score = new_score

    features_selected.remove(worst_feature)

    current_score = fisher_extraction(data[features_selected], classes)
    
    print(f"new worst feature: {worst_feature}, score: {best_score}")
    print(f"current features: {features_selected}, current score: {current_score}\n")


print(f"SBS finished, current features: {features_selected}\n")

fig = plt.figure()
plt.scatter(data[features_selected[0]], data[features_selected[1]], c=classes)
plt.title("Fisher + Plus-L Take-R")
#plt.show()