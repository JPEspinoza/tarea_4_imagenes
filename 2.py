"""
Este script hace el paso 2 de la tarea 4
2. A cada columna-característica, utilice la técnica media cero, desviación uno para normalizarla

El script toma el archivo 1.csv, lo pasa por StandardScaler y lo guarda en 2.csv
"""

from sklearn.preprocessing import StandardScaler
import pandas as pd

data = pd.read_csv("1.csv")

data = pd.DataFrame(StandardScaler().fit_transform(data), columns = data.columns)

data.to_csv("2.csv", index=False)