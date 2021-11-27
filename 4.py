"""

Este script entrena 3 clasificadores y los compara utilizando F-score

"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

classes = pd.read_csv("clase_letras.csv").to_numpy().ravel()

def score_models(data):

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

    score = f1_score(y_test, y_pred, average="micro")

    print(f"Naive Bayes F1-score: {score}")

    ################################################################
    # Arbol de Decision
    ################################################################

    decision_tree = DecisionTreeClassifier(criterion="entropy", random_state=2434)
    decision_tree.fit(X_train, y_train)

    y_pred = decision_tree.predict(X_test)

    score = f1_score(y_test, y_pred, average="micro")

    print(f"Decision Tree score: {score}")

    ################################################################
    # KNN
    ################################################################

    knn = KNeighborsClassifier(weights="distance", algorithm="ball_tree")

    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    score = f1_score(y_test, y_pred, average="micro")

    print(f"KNN score: {score}")

################################################################
################################################################
# Main program
################################################################
################################################################

# read data
data = pd.read_csv("features_scaled.csv")

# hu1 y hu2, los parametros preferidos de L10, R2
print("\nL10, R2: hu1 and hu2")
score_models(data[["hu1", "hu2"]])

# preferido de busqueda exhaustiva, que funciona peor que hu1 y hu2...
print("\nExhaustive search: all features")
score_models(data)

# LDA, que absolutamente destruye cualquier cosa que pueda hacer
print("\nLDA 3 features")
lda = LinearDiscriminantAnalysis(n_components=3)
lda_data = lda.fit_transform(data, classes)
score_models(lda_data)

# datos no estandarizados
data_not_scaled = pd.read_csv("features.csv")

# hu1 y roundness, elegidos sobre datos no escalados por L=10, R=2
print("\nL10, R2, not scaled, hu1 y roundness")
score_models(data_not_scaled[["hu1", "roundness"]])

# elegidos por L=10, R=5
print("\nL10, R5, not scaled, ['hu1', 'roundness', 'hu4', 'hu0', 'sonka1']")
score_models(data_not_scaled[['hu1', 'roundness', 'hu4', 'hu0', 'sonka1']])