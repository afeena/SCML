import numpy as np
import math
import parameter_estimator

def read_data(filename):
    X = []
    Y = []
    with open(filename) as train:
        k = int(next(train))
        d = int(next(train))
        while True:
            try:
                mat = [next(train) for x in range(17)]
                Y.append(int(mat[0]))
                fet = [[int(f) for f in x.split()] for x in mat[1:]]
                fet = np.ravel(fet)
                X.append(fet)
            except StopIteration:
                break
    return k, d, X, Y


def test(means, cov, priors):
    k, d, X, Y_true = read_data('data/usps.test')

    predicted = []
    dets = np.zeros(k)
    inversed_cov = np.zeros((k,d,d))
    for ki in range(k):
        det = np.linalg.slogdet(cov[ki])[1]
        dets[ki] = det
        cov_inv = np.linalg.inv(cov[ki])
        inversed_cov[ki] = cov_inv

    for x, y in zip(X, Y_true):
        pred_prob = []
        for ki in range(k):
            eex = -0.5*np.dot(np.dot(np.subtract(x, means[ki]), inversed_cov[ki]), np.subtract(x, means[ki]))
            dist = -(d/2)*np.log(2*np.pi)-0.5*dets[ki] + eex
            prob = np.log(priors[ki]) + dist
            pred_prob.append(prob)
        m=max(pred_prob)
        predicted.append(pred_prob.index(m)+1)

    conf_matrix = np.zeros((k, k))

    error = 0
    for pred_k, real_k in zip(predicted, Y_true):
        conf_matrix[pred_k - 1, real_k - 1] += 1
        if pred_k != real_k:
            error += 1

    err = error / len(X)
    print("empirical error rate: ", err)
    print("===confusion_matrix===")
    print(conf_matrix)
    print("======================")

def test_full_covariance(means, cov, priors):
    k, d, X, Y_true = read_data('data/usps.test')
    cov_inv = np.linalg.inv(cov)
    predicted = []
    for x, y in zip(X, Y_true):
        pred_prob = []
        for ki in range(k):
            eex = math.exp(
                -0.5 * (
                np.dot(np.dot(np.subtract(x, means[ki]), cov_inv), np.subtract(x, means[ki]))))
            prob = priors[ki] * eex

            pred_prob.append(prob)
        m=max(pred_prob)
        predicted.append(pred_prob.index(m)+1)

    conf_matrix = np.zeros((k, k))

    error = 0
    for pred_k, real_k in zip(predicted, Y_true):
        conf_matrix[pred_k - 1, real_k - 1] += 1
        if pred_k != real_k:
            error += 1

    err = error / len(X)
    print("empirical error rate: ", err)
    print("===confusion_matrix===")
    print(conf_matrix)
    print("======================")


def read_params():
    with open("parameters.txt") as param:
        tp = next(param)
        n_clases = int(next(param))
        dim = int(next(param))

        means = np.zeros((n_clases, dim))
        sigmas = np.zeros(dim)
        priors = np.zeros(n_clases)

        for x in range(n_clases):
            try:
                class_data = [next(param) for x in range(4)]
                priors[x] = class_data[1]
                means[x] = [float(m) for m in class_data[2].split()]
                sigmas = [float(s) for s in class_data[3].split()]

            except StopIteration:
                break
    return means, sigmas, priors


if __name__ == "__main__":

   # means, sigmas, priors = read_params()

    cs_diag_cov_mat, cs_cov_mat, pooled_diag_cov,pooled_full,means, priors = parameter_estimator.train()
    #test(means,cs_cov_mat, priors)
    l=[1, 0.5, 0.1, 0.01, 10e-3,10e-4,10e-5,10e-6]
    for ll in l:
        new_full_cov = np.zeros((10,256,256))
        for i,cov_k in enumerate(cs_cov_mat):
            n = ll*pooled_full+(1-ll)*cov_k
            new_full_cov[i,:,:] = n
        test(means, new_full_cov, priors)
    test_full_covariance(means, pooled_full, priors)
