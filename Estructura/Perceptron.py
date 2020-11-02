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
        #Valor de delta
        self.delta = 0

    #Funcion de Sigmoidal
    def SigmoidalFunction(self, x):
        return 1 / (1 + m.e**(-x))

    def activationFunction(self,x, type = None):
        if type is None:
            return self.SigmoidalFunction(x)
        
    #Funcion derivada
    def derivateFunction(self, x):
        return self.activationFunction(x)*(1 - self.activationFunction(x))

    #Calculo del valor propio del perceptron (F)
    def calculateValue(self):
        sum = 0
        for i in range( len(self.lastLayer.nodeslist) ):
            sum += self.weight[i] * self.activationFunction(self.lastLayer.nodeslist[i])
        return sum
    
    #Calcula la nueva proporcion de los pesos
    def calculateNewWeight(self, learningRate, expectedValue, nextLayer):
        newWeights = list()
        x = self.calculateValue()
        df = self.derivateFunction(x)
        #Si la capa del perceptron es una capa de salida
        if(nextLayer is None):
            fx = self.activationFunction(x)
            self.delta = (fx - expectedValue)*df
        
        else:
            #Si no es una capa output
            delta = 0.0
            nodesNextLayer = nextLayer.nodeslist
            for i in range( len(nodesNextLayer) ):
                node = nextLayer[i]
                deltax += node.delta * node.weight[i]
            self.delta = delta*df
         
        grad = self.delta * self.activationFunction(x)
        
        for i in range( len(self.weight)):
            newWeight = self.weight[i] - learningRate*grad    
            newWeights.append(newWeight)

        self.auxWeight = np.array(newWeights)

    #Funcion creada para actualizar los pesos de la red neuronal
    def updateWeight(self):
        self.weight = self.auxWeight
    
    def updateLastLayer(self, input):
        self.lastLayer = input