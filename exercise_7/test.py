from network import *

print("\nLoading Iris test data ")
testDataPath = "./data/irisTestData.txt"
testDataset = IrisDataset(testDataPath)


print("\nLoading Iris test data ")
trainDataPath = "./data/irisTrainData.txt"
trainDataset = IrisDataset(trainDataPath)


net = BasicNeuralNetwork()
net.train(trainDataset, testDataset)


