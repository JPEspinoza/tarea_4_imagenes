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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

data = pd.read_csv("2.csv")

# visualizar datos con PCA para ver clusters
# esto es SOLO para visualizar
pca = PCA(2)
pca_data = pca.fit_transform(data.drop('cluster', axis=1))
plt.figure()
plt.scatter(pca_data[:,0], pca_data[:,1], c=data[["cluster"]].to_numpy())

# visualizar datos con LDA
# esto fue lo que estamos intentando llegar en teoria
lda = LinearDiscriminantAnalysis(n_components=2)
lda_data = lda.fit_transform(data.drop('cluster', axis=1), data[["cluster"]])
plt.figure()
plt.scatter(lda_data[:,0], lda_data[:,1], c=data[["cluster"]].to_numpy())
plt.show()

# calcular fisher
def fisher_extraction(data, n_clusters, n_obs_per_cluster):

    # convierte a numpy array (matrix)
    data2 = data.to_numpy()

    # el codigo de abajo requiere un tensor?????
    x = n_obs_per_cluster+1
    y = len(data2[0])-1
    z = n_clusters

    G = np.zeros((x, y, z)) 

    for i in range(z):
        idx = (data2[:,-1]==i)

        G[:,:,i] = data2[idx,0:-1]
    
    # Buscamos la media
    D = G.reshape(-1,n_clusters)

    Vm = np.mean(D, axis=0)

    p = np.zeros((n_clusters,1))

    Vk = np.zeros((n_clusters,n_clusters))
    Gn = np.zeros((x, y, z))
    # Centrado
    for i in range(n_clusters):
        Vk[i,:] = np.mean(G[i,:,:], axis=0)-Vm         # centramos las medias
        Gn[i,:,:]= G[i,:,:]-repmat(Vm,y,1)   # centramos los puntos
        p[i] = len(G[i,:,:])/ D.shape[0]        # probabilidad de cada cluster

    # Inicialización
    Cb = np.zeros((z,z))
    Cw = np.zeros((z,z))

    for k in range(n_clusters):

        Cb = Cb + p[k]*np.matmul((Vk[k,:]-Vm).reshape(-1,1), (Vk[k,:]-Vm).reshape(1,-1))
        Cw = Cw + p[k]*np.cov(Gn[k,:,:].T)

    #Calculamos el índice de Fisher
    J = np.trace(np.matmul(np.linalg.inv(Cw),Cb))
    return (J)

fisher = fisher_extraction(data, 6, 5)

# do plus-l take-r
