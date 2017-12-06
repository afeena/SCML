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

def write_diagonal(fname,k,d,priors,means,sigmas, cs=True):
    with open(fname,"w") as p:
        p.write("d\n")
        p.write("{}\n".format(k))
        p.write("{}\n".format(d))
        i=0
        for pr,m in zip(priors,means):
            i+=1
            p.write("{}\n".format(i))
            p.write("{}\n".format(pr))
            p.write("{}\n".format(" ".join([str(x) for x in m.tolist()])))
            if cs:
                p.write("{}\n".format(" ".join([str(x) for x in sigmas[i-1].tolist()])))
            else:
                p.write("{}\n".format(" ".join([str(x) for x in sigmas.tolist()])))


def write_full(fname,k,d,priors,means,cov, cs=True):
    with open(fname,"w") as p:
        p.write("f\n")
        p.write("{}\n".format(k))
        p.write("{}\n".format(d))
        i=0
        for pr,m in zip(priors,means):
            i+=1
            p.write("{}\n".format(i))
            p.write("{}\n".format(pr))
            p.write("{}\n".format(" ".join([str(x) for x in m.tolist()])))
            if cs:
                for c in cov[i-1]:
                    p.write("{}\n".format(" ".join([str(x) for x in c.tolist()])))
            else:
                for cov_row in cov:
                    p.write("{}\n".format(" ".join([str(x) for x in cov_row.tolist()])))
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

    # pooled covariance
    res = np.zeros(d)
    res_full = np.zeros((d,d))
    for x, y in zip(X, Y):
        sub = np.subtract(x, means[y - 1])
        r = np.square(sub)
        res = np.add(res, r)
        m = sub*sub.reshape((d,1))
        res_full = np.add(res_full,m)
    pooled_diag_sigmas = np.divide(res, N)
    pooled_diag_cov = np.zeros((d,d))

    np.fill_diagonal(pooled_diag_cov,pooled_diag_sigmas)
    pooled_full = np.divide(res_full, N)


    #full class specific covariance matrix

    cs_cov_mat = np.zeros((k,d,d))
    cs_diag_cov_mat = np.zeros((k, d,d))
    sigmas_cs = np.zeros((k,d))
    for kn in range(k):
        res1 = np.zeros((d, d))
        res2 = np.zeros((1, d))
        k_ind = [i for i, y in enumerate(Y) if y == kn+1]
        k_subset = X[k_ind]

        for xn in k_subset:
            diff_vec = np.subtract(xn,means[kn])
            r = diff_vec*diff_vec.reshape(d,1)
            res1= np.add(res1,r)
            res2 = np.add(res2,np.square(diff_vec))
        cs_cov_mat[kn,:,:] = np.divide(res1,Nk[kn])
        sigmas_cs[kn,:] = np.divide(res2, Nk[kn])
        np.fill_diagonal(cs_diag_cov_mat[kn],np.divide(res2, Nk[kn]))


    priors = np.zeros(k)
    for i, nk in enumerate(Nk):
        priors[i] = nk / N


    write_full("full_pooled_cov.txt",k,d,priors,means,pooled_full, False)
    write_full("full_class_specific_cov.txt",k,d,priors,means,cs_cov_mat)
    write_diagonal('pooled_diagonal_cov.txt', k, d, priors, means, pooled_diag_sigmas, False)
    write_diagonal('class_specific_diagonal_cov.txt',k, d, priors, means, sigmas_cs)

    return cs_diag_cov_mat, cs_cov_mat, pooled_diag_cov, pooled_full,means, priors

if __name__=="__main__":
    train()