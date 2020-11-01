import numpy as np

def loadData(nameFile):
    file = open(nameFile,'r')
    matrix = np.loadtxt(file, delimiter=',', skiprows=1)
    return matrix

def main():
    nameDate = 'Dataset/fashion-1.csv'
    
    return 