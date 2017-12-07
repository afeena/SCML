from network import *

print("\nLoading Iris test data ")
testDataPath = "./data/irisTestData.txt"
testDataset = IrisDataset(testDataPath)


print("\nLoading Iris train data ")
trainDataPath = "./data/irisTrainData.txt"
trainDataset = IrisDataset(trainDataPath)


net = BasicNeuralNetwork(layer_sizes=[5], num_input=4, num_output=3, num_epoch=500, learning_rate=0.001,
                 mini_batch_size=8)
net.train(trainDataset, testDataset)
