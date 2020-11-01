from Perceptron import perceptron
from layer import layer

import copy as cp
import numpy as np

class red:
    #Se inicializa la red
    def __init__(self):
        # Capaz de la red neuronal
        self.red : list()
        # Cantidad de capaz de la red
        self.nLayer = 0
    
    #Se agrega una capa nueva
    def addLayer(self, nodeSize, idLayer):
        newLayer = layer(idLayer, nodeSize, self.red[-1])    
        self.red.append( newLayer )
        self.nLayer+=1

    #Retorna el vector predecido
    def predictVector(self, input):
        predicted = None
        count = 0
        #Se actualiza el valor de todas las capas
        for l in self.red:
            l.evaluateLayer()
            #Si estamos en la ultima capa, retornamos el valor de la capa
            if(count == self.nLayer):
                predicted = l.getLayerValue()
            else:
                count +=1
                
        return predicted


