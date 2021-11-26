"""

Este script entrena 3 clasificadores y los compara utilizando F-score

"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# read data
data = pd.read_csv("2.csv")

# put classes on np array
# drop classes from data
classes = data[["cluster"]].to_numpy().ravel()
data = data.drop("cluster", axis=1)

### prepare training data
# features to use, learned with script 3
features = ["hu1", "hu2"]
data = data[features].to_numpy()

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