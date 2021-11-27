"""
bonus: Predecir y marcar la sopa de letras

usamos LDA porque francamente desintegra todos los demas metodos

haciendo muchas pruebas en 4 usualmente vi el mejor rendimiento con
decision tree, asi que uso ese.

nos ponemos algunos desafios extra
entrenaremos con tan solo el 50% de los datos

para no tener que volver a extraer y normalizar los datos
sencillamente explotamos el hecho de que regionprops siempre
encuentra en el mismo orden, asi que usamos los datos
que ya tenemos procesados y solo usamos la imagen para
mostrar las clasificaciones visualmente
"""

import cv2
import pandas as pd
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# leemos los datos
classes = pd.read_csv("clase_letras.csv").to_numpy().ravel()
data = pd.read_csv("features_scaled.csv")

# hacemos LDA
lda = LinearDiscriminantAnalysis(n_components=3)
lda_data = lda.fit_transform(data, classes)

plt.figure()
plt.scatter(lda_data[:,0], lda_data[:,1], c=classes)

# entrenamos un modelo
# usamos solo el 50% de los datos
X_train, X_test, y_train, y_test = train_test_split(lda_data, classes, train_size=0.5)

decision_tree = DecisionTreeClassifier(criterion="entropy")
decision_tree.fit(X_train, y_train)

predictions = decision_tree.predict(lda_data)

print("Score")
print(decision_tree.score(X_test, y_test))
print()

### leemos la imagen
image = cv2.imread("sopa_letras.png")
greyscale_image =  cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, binary_image = cv2.threshold(greyscale_image, 125,255,cv2.THRESH_BINARY_INV)

letras ={0:'a', 1:'b', 2:'c', 3:'e', 4:'f', 5:'g'}

#find regions
labels = label(binary_image)

# plot characters in the image
plt.figure()
plt.imshow(labels, cmap='jet')

for i, region in zip(range(len(predictions)), regionprops(labels)):
    min_row, min_col, max_row, max_col = region.bbox

    plt.text(min_col,  min_row-5, f'{letras[predictions[i]]}', c="yellow")

plt.show()