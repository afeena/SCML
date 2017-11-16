import numpy as np

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



def train():
    k, d, X, Y = read_data('data/usps.train')
    means = np.zeros((k, d))
    Nk = np.zeros(k)
    for xi, yi in zip(X, Y):
        means[yi - 1] = np.add(means[yi - 1], xi)
        Nk[yi - 1] += 1
    for i, m in enumerate(means):
        means[i] = np.divide(m, Nk[i])
    N = len(X)

    #Diagonal pooled covariance
    res = np.zeros(d)
    res_fill = np.zeros((d,d))
    for x, y in zip(X, Y):
        sub = np.subtract(x, means[y - 1])
        r = np.square(sub)
        res = np.add(res, r)
        m = sub*sub.reshape((d,1))
        res_fill = np.add(res_fill,m)
    pooled_diag_sigmas = np.divide(res, N)
    pooled_diag_cov = np.zeros((d,d))

    np.fill_diagonal(pooled_diag_cov,pooled_diag_sigmas)
    pooled_full = np.divide(res_fill, N)


    #full class specific covariance matrix
    res1 = np.zeros((d,d))
    res2 = np.zeros((1,d))
    cs_cov_mat = np.zeros((k,d,d))
    cs_diag_cov_mat = np.zeros((k, d,d))
    for kn in range(k):
        k_ind = [i for i, y in enumerate(Y) if y == kn+1]
        k_subset = X[k_ind]
        for xn in k_subset:
            diff_vec = np.subtract(xn,means[kn])
            r = diff_vec*diff_vec.reshape(d,1)
            res1 = np.add(res1,r)
            res2 = np.add(res2,np.square(diff_vec))
        cs_cov_mat[kn,:,:] = np.divide(res1, Nk[kn])
        np.fill_diagonal(cs_diag_cov_mat[kn],np.divide(res2, Nk[kn]))

    priors = np.zeros(k)
    for i, nk in enumerate(Nk):
        priors[i] = nk / N

    return cs_diag_cov_mat, cs_cov_mat, pooled_diag_cov, pooled_full,means, priors

    with open("parameters.txt","w") as p:
        p.write("d\n")
        p.write("{}\n".format(k))
        p.write("{}\n".format(d))
        i=0
        for pr,m in zip(priors,means):
            i+=1
            p.write("{}\n".format(i))
            p.write("{}\n".format(pr))
            p.write("{}\n".format(" ".join([str(x) for x in m.tolist()])))
            p.write("{}\n".format(" ".join([str(x) for x in pooled_diag_sigmas.tolist()])))
    print("results in parameters.txt",)

if __name__=="__main__":
    train()