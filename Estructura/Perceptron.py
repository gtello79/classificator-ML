from layer import layer
import numpy as np
import math as m
import copy as cp

#Perceptron
class perceptron:
    #Inicializacion del perceptron
    def __init__(self, lastLayer, weight, id):
        #id perceptron
        self.id_layer = id
        # Valor del nodo
        self.valueState = 0.0
        #Nodos de la capa anterior 
        self.lastLayer = cp.deepcopy(lastLayer.getLayerValue) 
        #Pesos asignados con las neuronas de la capa anterior
        self.weight : weight
        #Vector auxiliar para pesos
        self.auxWeight: list()
        #Vector neuronas next Layer

    #Funcion de activacion
    def activationFunction(self, x):
        return 1 / (1 + m.e**(-x))
    
    #Funcion derivada
    def derivateFunction(self, x):
        return self.activationFunction(x)*(1 - self.activationFunction(x))

    
    #Calculo del valor propio del perceptron
    def calculateValue(self):
        sum = 0
        for i in range( len(self.lastLayer.nodeslist) ):
            sum += self.weight[i] * self.activationFunction(self.lastLayer.nodeslist[i])
        self.valueState = sum
    

    #Calcula la nueva proporcion de los pesos
    def calculateNewWeight(self, learningRate, expectedValue, id_layer):
        delta = []
        delta_i = 0
        
        #Si la capa del perceptron es una capa de salida
        if(self.id_layer == id_layer):
            for w in self.weight:
                delta_i = (self.activationFunction(w) - expectedValue) * self.derivateFunction(w)
                delta.append(delta_i)
        else:
        #Si no es capa de salida
            for w in self.weight:
                delta_i = (self.activationFunction(w) * w) * self.derivateFunction(w)
                delta.append(delta_i)
        
        delta =  np.array(delta)

    