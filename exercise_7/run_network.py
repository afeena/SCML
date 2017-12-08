from network import *
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--train_data',
                    help='path to train data', default='./data/irisTrainData.txt')
parser.add_argument('--test_data',
                    help='path to test data', default="./data/irisTestData.txt")


args = parser.parse_args()


print("\nLoading Iris test data ")
testDataPath = args.test_data
testDataset = IrisDataset(testDataPath)


print("\nLoading Iris train data ")
trainDataPath = args.train_data
trainDataset = IrisDataset(trainDataPath)


net = BasicNeuralNetwork(layer_sizes=[5], num_input=4, num_output=3, num_epoch=50, learning_rate=0.1,
                 mini_batch_size=8)
net.train(trainDataset, eval_dataset=testDataset)
