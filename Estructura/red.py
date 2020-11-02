from layer import layer

import numpy as np

class red:
    #Se inicializa la red
    def __init__(self, layerSize, nodesSize):
        # Capaz de la red neuronal
        self.red : list()
        # Cantidad de capaz de la red
        self.nLayer = 0
        #Cantidad de capaz en la red
        self.layerSize = layerSize
        #Cantidad de nodos x cada capa
        self.nodesSize = nodesSize
        #Se llama a la inicializacion de la red
        self.initializeWed()

    #Se inicializa la red creando cada una de sus capaz
    def initializeWed(self):
        for l in range( len(self.layerSize) ):
            self.addLayer(self.nodesSize, l)
    
    #Se agrega una capa nueva
    def addLayer(self, nodeSize, idLayer):
        newLayer = layer(idLayer, nodeSize, self.red[-1])    
        self.red.append( newLayer )
        self.nLayer+=1

    #Retorna el vector predecido
    def predictVector(self, input):
        #Se actualiza el valor de todas las capas
        for i in range(len(input)):
            self.red[0].nodesList[i].value = input[i]    
        layer = self.red[-1]
        predicted = layer.getLayerValue()

        return predicted

    #Se calculan los pesos de la red
    def calculateNewWeights(self, learningRate, expectedValue):
        for i in reversed( range(len(self.red)) ):
            actual_layer = self.red[i]
            #Se verifica el tipo de la capa actual
            if( i == len(self.red)-1 ):
                #Capa de salida
                actual_layer.calculateNewWeight(learningRate,expectedValue)
            else:
                #Capa intermedia o input
                nextLayer = self.red[i+1]
                actual_layer.calculateNewWeight(learningRate,expectedValue, nextLayer)

    #Entrenamiento de la red neuronal
    def training(self, instances, expectedValues, numIteration, learningRate):
        instance = instances
        expectedValue = expectedValues
        #insertar el vector de input y evaluar los nodos
        self.insertInput(instance)
        for l in self.red:
            l.evaluateLayer()

        #Obtener predicción
        v = self.predictVector(instances)

        #Calcular Pesos
        self.calculateNewWeights(learningRate, expectedValue)

        #Actualizar los pesos
        for l in self.red:
            l.updateWeights()

    def insertInput(self, input):
        self.red[0].insertInput(input)

            


