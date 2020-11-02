from Estructura.red import red
import numpy as np


def loadData(nameFile):
    file = open(nameFile,'r')
    matrix = np.loadtxt(file, delimiter=',', skiprows=1)
    return matrix

def main():
    nameDate = 'Dataset/fashion-1.csv'
    instances = loadData(nameDate)
    testInstances = 4
    validationInstances = 4
    layerSize = 3
    nodesSize = [784,20,10]
    learningRate = 0.01
    

    neuronalNet = red(layerSize,nodesSize)
    