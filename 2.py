"""
Este script hace el paso 2 de la tarea 4
2. A cada columna-característica, utilice la técnica media cero, desviación uno para normalizarla

Este script hace el paso dos, escalando los datos por standard scaler
Los descriptores de fourier son escalados a mano, restandoles el promedio y dividiendo por desviacion
Tambien se le agregan manualmente los clusters para ser usados en los scripts posteriores
"""

from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# we index every character in the list
# this will serve as the clusters

# leemos los datos
# pandas no sabe manejar datos complejos asi que los convertimos manualmente
data = pd.read_csv(
    "features.csv", 
    converters={
        'fourier0': lambda s: complex(s),
        'fourier1': lambda s: complex(s),
        'fourier2': lambda s: complex(s),
        'fourier3': lambda s: complex(s),
        'fourier4': lambda s: complex(s),
    }
)

# extraemos fourier para normalizar aparte, ya que 
# standard scaler no sabe usar datos complejos
fourier = data[[
    "fourier0", 
    "fourier1", 
    "fourier2", 
    "fourier3", 
    "fourier4", 
]]

# normalizamos restando promedio y dividiendo por desv est.
fourier = (fourier - fourier.mean()) / fourier.std()

# separamos en datos reales e imaginarios
fourier_split = pd.DataFrame()

fourier_split["fourier0_real"] = np.real(fourier['fourier0'])
fourier_split["fourier0_imag"] = np.imag(fourier['fourier0'])

fourier_split["fourier1_real"] = np.real(fourier['fourier1'])
fourier_split["fourier1_imag"] = np.imag(fourier['fourier1'])

fourier_split["fourier2_real"] = np.real(fourier['fourier2'])
fourier_split["fourier2_imag"] = np.imag(fourier['fourier2'])

fourier_split["fourier3_real"] = np.real(fourier['fourier3'])
fourier_split["fourier3_imag"] = np.imag(fourier['fourier3'])

fourier_split["fourier4_real"] = np.real(fourier['fourier4'])
fourier_split["fourier4_imag"] = np.imag(fourier['fourier4'])

# extraemos todos los demas datos, excluyendo fourier que ya trabajamos
data = data.drop(
    labels=[
        "fourier0", 
        "fourier1", 
        "fourier2", 
        "fourier3", 
        "fourier4", 
    ],
    axis=1
)

# escalamos los datos
# los ponemos en un dataframe con las mismas columnas que el original
data = pd.DataFrame(StandardScaler().fit_transform(data), columns = data.columns)

# agregamos fourier
data = data.join(fourier_split)

data.to_csv("features_scaled.csv", index=False)