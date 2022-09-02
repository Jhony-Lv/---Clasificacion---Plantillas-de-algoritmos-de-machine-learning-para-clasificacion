# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 23:19:29 2020

@author: Jhony Lv
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Importamos el dataset
dataset = pd.read_csv()

x = dataset.iloc[ : , ].values
y = dataset.iloc[ : , ].values


#Codificamos los datos categoricos (En caso de haberlos)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

datos_categoricos = ColumnTransformer(transformers = [
        ('---', OneHotEncoder(drop = 'first'), []),
        ('---', OneHotEncoder(drop = 'first'), [])],
         remainder = 'passthrough')

x = datos_categoricos.fit_transform(x)
x = np.array(x, dtype=np.float64)


#Dividimos el dataset en conjunto de entrenamiento y testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

#Realizamos el escalado de variables
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)


#Parte 2 - Construir la RNA

#Importar Keras y librerías adicionales
import keras
from keras.models import Sequential
from keras.layers import Dense


#Inicializar la RNA
classifier = Sequential()

#Añadimos las capas de entrada y la primera capa oculta 
classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu", input_dim = 11))

#Añadir la segunda capa oculta
classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu"))

#Añadir la capa final del a RNA
classifier.add(Dense(units = 1, kernel_initializer = "uniform", activation = "sigmoid"))

#Compilar la RNA
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

#Ajustamos la RNA al conjuto de entrenamiento 
classifier.fit(x_train, y_train,  batch_size = 10, epochs = 20)


# Parte 3 - Evaluar el modelo y calcular las predicciones finales 

#Predicción de los resultados con el conjunto de testing
y_pred = classifier.predict(x_test)
#y_pred = (y_pred > 0.5)

#Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)










