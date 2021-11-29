"""

Este script entrena 3 clasificadores y los compara utilizando F-score

"""

import matplotlib.pyplot as plt
from numpy.lib.function_base import disp
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

classes = pd.read_csv("clase_letras.csv").to_numpy().ravel()

def score_models(data, display=True):

    try:
        # convertir a np array si no lo es
        # para que no tire warning
        data = data.to_numpy()
    except:
        pass

    # create training splits
    X_train, X_test, y_train, y_test = train_test_split(data, classes, train_size=0.7, random_state=2434)

    ################################################################
    # Naive Bayes
    ################################################################

    naive_bayes = GaussianNB()
    naive_bayes.fit(X_train, y_train)

    y_pred = naive_bayes.predict(X_test)

    nb_score = f1_score(y_test, y_pred, average="micro")
    if display:
        print(f"Naive Bayes F1 score: {nb_score:,.2f}")

    ################################################################
    # Arbol de Decision
    ################################################################

    decision_tree = DecisionTreeClassifier(criterion="entropy", random_state=2434)
    decision_tree.fit(X_train, y_train)

    y_pred = decision_tree.predict(X_test)

    dt_score = f1_score(y_test, y_pred, average="micro")

    if display:
        print(f"Decision Tree F1 score: {dt_score:,.2f}")

    ################################################################
    # KNN
    ################################################################

    knn = KNeighborsClassifier(weights="distance", algorithm="ball_tree")

    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    knn_score = f1_score(y_test, y_pred, average="micro")

    if display:
        print(f"KNN F1 score: {knn_score:,.2f}")

    return (nb_score, dt_score, knn_score)

################################################################
################################################################
# Main program
################################################################
################################################################

# read data
data = pd.read_csv("features_scaled.csv")

print("\nL2, R2: ['hu1', 'hu2']")
score_models(data[['hu1', 'hu2']])

print("\nL5, R2: ['hu1', 'sonka0']")
score_models(data[['hu1', 'sonka0']])

print("\nL10 a 20, R2: ['sonka0', 'hu0']")
score_models(data[['sonka0', 'hu0']])

print("\nL5, R3: ['hu1', 'sonka0', 'hu3']")
score_models(data[['hu1', 'sonka0', 'hu3']])

print("\nL10 a 20, R3: ['sonka0', 'roundness', 'hu0']")
score_models(data[['sonka0', 'roundness', 'hu0']])

print("\nL5, R4:  ['hu1', 'sonka0', 'hu3', 'fourier2_real']")
score_models(data[['hu1', 'sonka0', 'hu3', 'fourier2_real']])

print("\nL10 a 20, R4: ['sonka0', 'hu3', 'roundness', 'hu0']")
score_models(data[['sonka0', 'hu3', 'roundness', 'hu0']])

print("\nL10, R10:['hu2', 'sonka0', 'hu3', 'fourier2_imag', 'sonka2', 'roundness', 'hu0', 'hu4', 'hu6', 'fourier1_real']")
score_models(data[['hu2', 'sonka0', 'hu3', 'fourier2_imag', 'sonka2', 'roundness', 'hu0', 'hu4', 'hu6', 'fourier1_real']])

print("\nL20, R10:['hu1', 'sonka0', 'hu3', 'fourier2_imag', 'sonka2', 'roundness', 'hu0', 'hu4', 'hu5', 'fourier3_real']")
score_models(data[['hu1', 'sonka0', 'hu3', 'fourier2_imag', 'sonka2', 'roundness', 'hu0', 'hu4', 'hu5', 'fourier3_real']])

# preferido de busqueda exhaustiva, que funciona peor que hu1 y hu2...
print("\nExhaustive search: all features")
score_models(data)

### datos no estandarizados
data_not_scaled = pd.read_csv(
    "features.csv",
    converters={
        'fourier0': lambda s: np.real(complex(s)),
        'fourier1': lambda s: np.real(complex(s)),
        'fourier2': lambda s: np.real(complex(s)),
        'fourier3': lambda s: np.real(complex(s)),
        'fourier4': lambda s: np.real(complex(s)),
    }
)

print("\nL5, R2, not scaled,['roundness', 'hu0']")
score_models(data_not_scaled[['roundness', 'hu0']])

print("\nL10, R2, not scaled,['hu0', 'sonka0']")
score_models(data_not_scaled[["hu0", "sonka0"]])

print("\nL10, R5, not scaled,['roundness', 'hu0', 'sonka0', 'hu5', 'hu3']")
score_models(data_not_scaled[['roundness', 'hu0', 'sonka0', 'hu5', 'hu3']])

print("\nL20, R5, not scaled,['roundness', 'hu0', 'sonka0', 'sonka2', 'hu3']")
score_models(data_not_scaled[['roundness', 'hu0', 'sonka0', 'sonka2', 'hu3']])

################
# graficar

# el orden fue conseguido solo por forward search
# o sea, se hizo plus-L take-R con L=23 y R=23
feature_order = ['hu1', 'hu2', 'sonka0', 'hu3', 'fourier2_real', 'fourier4_imag', 'fourier2_imag', 'sonka2', 'roundness', 'hu0', 'hu4', 'hu6', 'fourier1_real', 'complexity', 'hu5', 'fourier3_real', 'fourier3_imag', 'fourier0_real', 'sonka3', 'fourier4_real', 'fourier0_imag', 'fourier1_imag', 'sonka1']

current_features = []
full_scores = []

for feature in feature_order:
    current_features.append(feature)
    full_scores.append(score_models(data[current_features], False))

full_scores = np.array(full_scores)

plt.figure()
plt.plot(range(1, len(feature_order)+1), full_scores[:,0])
plt.title("Naive Bayes")
plt.xlabel("Numero de features")
plt.ylabel("F1-Score")

plt.figure()
plt.plot(range(1,len(feature_order)+1), full_scores[:,1])
plt.title("Decision Tree")
plt.xlabel("Numero de features")
plt.ylabel("F1-Score")

plt.figure()
plt.plot(range(1,len(feature_order)+1), full_scores[:,2])
plt.title("KNN")
plt.xlabel("Numero de features, datos no escalados")
plt.ylabel("F1-Score")

plt.show()