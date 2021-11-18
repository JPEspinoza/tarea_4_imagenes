"""
Este script hace el paso 2 de la tarea 4
2. A cada columna-característica, utilice la técnica media cero, desviación uno para normalizarla

El script toma el archivo 1.csv, lo pasa por StandardScaler y lo guarda en 2.csv
"""

from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# leemos el csv
# pandas no sabe leer numeros complejos, asi que los convertimos manualmente
data = pd.read_csv(
    "1.csv", 
    converters={
        'fourier0': lambda s: complex(s),
        'fourier1': lambda s: complex(s),
        'fourier2': lambda s: complex(s),
        'fourier3': lambda s: complex(s),
        'fourier4': lambda s: complex(s),
    }
)

# extraemos fourier para normalizar aparte
fourier = data[[
    "fourier0", 
    "fourier1", 
    "fourier2", 
    "fourier3", 
    "fourier4", 
]]

# normalizamos restando promedio y dividiendo por desv est.
fourier = (fourier - fourier.mean()) / fourier.std()

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

# fusionamos el dataframe de fourier que escalamos a mano y el nuevo
data = data.join(fourier)

data.to_csv("2.csv", index=False)