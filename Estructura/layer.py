from Estructura.Perceptron import perceptron
import random as rd
import numpy as np

class layer:
    #Se inicia la clase capa
    def __init__(self, idLayer, nodesSize):
        #Id de la capa, para identificar si es capa oculta u otra
        self.idLayer = idLayer
        
        #Copia de la capa anterior
        self.lastLayer = None
        
        #Listado de nodos de la capa
        self.nodesList = list()
    
        #Se inicializa la capa
        self.initializeLayer(nodesSize)


    #Inicializacion de la capa
    def initializeLayer(self, nodesSize):
        nodes = list()
        #Se agregan la cantidad de nodos asociada
        for n in range(nodesSize): 
            weight = self.initializeweight(nodesSize)
            node =  perceptron(self.lastLayer, weight, n)
            nodes.append(node) 
        self.nodesList = np.array(nodes)

    #Calcula el valor de los nuevos pesos para cada nodo
    def calculateNewWeight(self, learningRate, expectedValue, nextLayer = None):
        for n in self.nodesList:
            n.calculateNewWeight(learningRate, expectedValue, nextLayer)

    def updateWeights(self):
        for n in self.nodesList:
            n.updateweights()

    #Se inicializan pesos aleatorios para un total de 'nodeSize'
    def initializeweight(self, nodesSize):
        #Se agregan los pesos random
        weight = list()
        for i in range(nodesSize):
            x = rd.uniform(0,1)
            weight.append(x)
        #Retorna un vector con pesos aleatorioss
        return np.array( weight )

    #Calcular el valor de cada nodo de la capa
    def evaluateLayer(self):
        for n in self.nodesList:
            n.calculateValue()

    #Retorna un arreglo con el valor de todos los nodos    
    def getLayerValue(self):
        #Retorna el valor de la capa
        val = []
        for n in self.nodesList:
            val.append( n.valueState )
        return np.array(val)

    #Insertar un vector de entrada
    def insertInput(self,inputVector):
        for i in range( len(inputVector) ):
            self.nodesList[i].updateLastLayer( inputVector[i] )
