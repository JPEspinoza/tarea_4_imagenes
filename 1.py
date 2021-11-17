"""
Este script hace el paso 1 de la tarea 4.
1. Extraccion de las siguientes caracteristicas de cada region encontrada
    1. redondez
    2. momentos de hu
    3. sonka, hlavac y boyle
    4. descriptores de fourier
    5. complejidad de forma
Tras extraer las caracteristicas exporta las caracteristicas en 1.csv
"""

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage.fourier import fourier_uniform
from skimage.measure import label, regionprops, EllipseModel
from matplotlib import pyplot as plt

def sonka_moments(m):
    """
    Receives a 3x3 moment matrix and returns tuple of 4 Sonka, Hlavac and Boyle moments
    this was not fun to implement
    """

    i1 = (m[2,0]*m[0,2]-m[1,1]**2)/m[0,0]**4

    i2 = (
        m[3,0]**2*m[0,3]**2
        -6*m[3,0]*m[2,1]*m[1,2]*m[0,3]
        +4*m[3,0]*m[1,2]**3
        +4*m[2,1]**3*m[0,3]
        -3*m[2,1]**2*m[1,2]**2
    ) / m[0,0]**10 

    i3 = (
        m[2,0]*(m[2,1]*m[0,3]-m[1,2]**2)
        -m[1,1]*(m[3,0]*m[0,3]-m[2,1]*m[1,2])
        +m[0,2]*(m[3,0]*m[1,2]-m[2,1]**2)
    ) /m[0,0]**7

    # sometimes m01 is 0, we can't do a division by zero so just
    # set sonka4 to 0 instead
    # hu6 does the same thing as well
    if(m[1,0] != 0):
        i4 = (
            m[2,0]**3*m[0,3]**2
            -6*m[2,0]**2*m[1,1]*m[1,2]*m[0,3]
            -6*m[2,0]**2*m[0,2]*m[2,1]*m[0,3]
            +9*m[2,0]**2*m[0,2]*m[1,2]**2
            +12*m[2,0]*m[1,1]**2*m[2,1]*m[0,3]
            +6*m[2,0]*m[1,1]*m[0,2]*m[3,0]*m[0,3]
            -18*m[2,0]*m[1,1]*m[0,2]*m[2,1]*m[1,2]
            -8*m[1,1]**3*m[3,0]*m[0,3]
            -6*m[2,0]*m[0,2]**2*m[3,0]*m[1,2]
            +9*m[2,0]*m[0,2]**2*m[2,1]
            +12*m[1,1]**2*m[0,2]*m[3,0]*m[1,2]
            -6*m[1,1]*m[0,2]**2*m[3,0]*m[2,1]
            +m[0,2]**3*m[3,0]**2
        ) / m[1,0] ** 11
    else: 
        i4 = 0

    return i1, i2, i3, i4

texture_features = ['contrast','correlation', 'dissimilarity','homogeneity','ASM','energy']

image = cv2.imread("sopa_letras.png")

greyscale_image =  cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, binary_image = cv2.threshold(greyscale_image, 125,255,cv2.THRESH_BINARY_INV)

data = []

#find regions
labels = label(binary_image)

#loop over regions
for region in regionprops(labels):
    ### get hu
    hu = region.moments_hu

    ### sonka moments
    sonka = sonka_moments(region.moments_central)
    #print(sonka)

    ####
    ### shape complexity
    #### 
    # countours
    contours, _ = cv2.findContours(
        np.array(region.image, dtype='uint8'), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_NONE
    )

    contours = np.vstack(contours[0])

    # mean distance
    x, y= zip(*contours)
    x= np.array(x).reshape(-1,1)
    y= np.array(y).reshape(-1,1)
    centroid_y, centroid_x = region.centroid
    distances = np.sqrt((x-centroid_x)**2+(y-centroid_y)**2)
    mean_distance = np.mean(distances)

    # finally, complexity
    complexity = (region.area) / (mean_distance ** 2)

    ####
    ### fourier
    ####
    x, y = zip(*contours)
    complex_coords = np.array(x) + 1j*np.array(y)
    complex_coords = np.append(complex_coords, complex_coords[0:1])

    fourier = np.fft.fft(complex_coords)
    #Energía
    E = np.abs(fourier)/np.sum(np.abs(fourier))
    acum = np.cumsum(E)

    #ordenamos de menor a mayor
    idx = np.argsort(E)
    idx = idx[::-1] #orden inverso

    fourier = fourier[idx[0:5]]

    ### perimeter
    # perimeter with line over center of pixes
    # connectivity 4
    perimeter = region.perimeter

    ### roundness:
    roundness = (4 * region.area * np.pi) / (region.perimeter ** 2)

    # add data to dataframe
    data.append(
        {
            "roundness": roundness, 
            "hu0": hu[0], 
            "hu1": hu[1], 
            "hu2": hu[2], 
            "hu3": hu[3], 
            "hu4": hu[4], 
            "hu5": hu[5], 
            "hu6": hu[6], 
            "sonka0": sonka[0],
            "sonka1": sonka[1],
            "sonka2": sonka[2],
            "sonka3": sonka[3],
            "fourier0": fourier[0],
            "fourier1": fourier[1],
            "fourier2": fourier[2],
            "fourier3": fourier[3],
            "fourier4": fourier[4],
            "complexity": complexity,
        }
    )

df = pd.DataFrame(data)

df.to_csv("1.csv", index=False)