import numpy as np
from sklearn import mixture as mxt
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
                fet = [[int(f)  for f in x.split()] for x in mat[1:]]
                fet = np.ravel(fet)
                X.append(fet)
            except StopIteration:
                break
    return k, d, np.array(X), np.array(Y)

def test(means, cov, priors):
    k, d, X, Y_true = read_data('data/usps.test')

    predicted = []
    dets = []
    inversed_cov = np.zeros((k,d,d))
    for ki in range(k):
        det = np.linalg.slogdet(cov[ki])
        dets.append(det)
        cov_inv = np.linalg.inv(cov[ki])
        inversed_cov[ki] = cov_inv

    for x, y in zip(X, Y_true):
        pred_prob = []
        for ki in range(k):
            eex = -0.5*np.dot(np.dot(np.subtract(x, means[ki]), inversed_cov[ki]), np.subtract(x, means[ki]))
            dist = -0.5*d*np.log(2*np.pi)-0.5*dets[ki][0]*dets[ki][1] + eex
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

k, d, X_train, Y_train = read_data('data/usps.train')

model1 = mxt.GaussianMixture(n_components=k)
model1.fit(X_train,Y_train)
e_k = model1.covariances_

means = model1.means_
model2 = mxt.GaussianMixture(n_components=k,covariance_type='tied').fit(X_train,Y_train)
e = model2.covariances_

model3 = mxt.GaussianMixture(n_components=k,covariance_type='diag').fit(X_train,Y_train)
ed = model3.covariances_
k, d, X_test, Y_test = read_data('data/usps.test')

predicted = model2.predict(X_test)
print(1)


#test(means,cs_cov_mat, priors)
# l=[1, 0.9, 0.5, 0.1, 0.01, 10e-3,10e-4,10e-5,10e-6]
# for ll in l:
#     new_full_cov = np.zeros((10,256,256))
#     for i,cov_k in enumerate(e_k):
#         n = ll*e+(1-ll)*cov_k
#         new_full_cov[i,:,:] = n
#     print("lambda: ",ll)
# test(means, e_k, np.random.uniform(0,1,10))

