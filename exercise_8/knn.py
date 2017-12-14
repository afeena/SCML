import numpy as np
import argparse
from collections import Counter
class KNN:
    def __init__(self,k):
        self.k = k
        self.obs = None
        self.classes = None
        self.proirs = None


    def train(self,fn):
        Nk,self.obs,self.classes = self.read_data(fn)
        N = len(self.obs)
        self.proirs = [0 for kn in range(Nk)]
        for kn in range(Nk):
            k_ind = [i for i,c in enumerate(self.classes) if c==kn]
            self.proirs[kn] = len(self.obs[k_ind])/N

    def test(self, fn):
        Nk, Xt, Yt = self.read_data(fn)
        error = 0
        c = self.classify_matrix(Xt)
        error = sum([1 for t,cn in zip(Yt,c) if t!=cn])

        #     if c!=yn:
        #         error+=1
        #         print(error)
        test_error = error/len(Xt)
        print(test_error)

    def classify_one(self, v):
        result = np.zeros((1,))
        d = np.linalg.norm(self.obs - xn, axis=1) * [self.proirs[yn] for yn in self.classes]
        sorted_ind = np.argsort(d)[1:self.k]
        c = Counter(self.classes[sorted_ind])
        res = c.most_common(1)[0][0]
        result[0] = res
        return result

    def classify_matrix(self, X):
        result = np.zeros((X.shape[0],))
        for i,xn in enumerate(X):
            d = np.linalg.norm(self.obs - xn, axis=1)*[self.proirs[yn] for  yn in self.classes]
            sorted_ind = np.argsort(d)[1:self.k]
            c = Counter(self.classes[sorted_ind])
            res = c.most_common(1)[0][0]
            result[i] = res
        return result

    def read_data(self, filename):
        X = []
        Y = []
        with open(filename) as train:
            k = int(next(train))
            d = int(next(train))
            while True:
                try:
                    mat = [next(train) for x in range(17)]
                    Y.append(int(mat[0])-1)
                    fet = [[int(f)  for f in x.split()] for x in mat[1:]]
                    fet = np.ravel(fet)
                    X.append(fet)
                except StopIteration:
                    break
        return k, np.array(X), np.array(Y)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('k',
                        help='number of k-neighbours', default=1, type=int)
    parser.add_argument('--train_data',
                        help='path to train data', default='./data/usps.train')
    parser.add_argument('--test_data',
                        help='path to test data', default="./data/usps.test")


    args = parser.parse_args()

    k = args.k
    knn = KNN(k)
    knn.train(args.train_data)
    knn.test(args.test_data)