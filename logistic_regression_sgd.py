#!/usr/bin/env python

import numpy as np
import scipy.special as sps
import matplotlib.pyplot as plt
import data_utils as a2

max_iter = 500
tol = 0.00001
eta = [0.5, 0.3, 0.1, 0.05, 0.01]

data = np.genfromtxt('data.txt')
X = data[:,0:3]
t = data[:,3]

e_all = [[] for x in range(len(eta))]
np.random.seed(1202)
for i in range(len(eta)):
    w = np.array([0.1, 0, 0])
    for iter in range (0,max_iter):
        e_temp = []
        for j in np.random.choice(X.shape[0],size=X.shape[0],replace=False):
            y = sps.expit(np.dot(X[j],w))
            e = -(np.multiply(t[j],np.log(y+0.00001)) + np.multiply((1-t[j]),np.log(1-y+0.00001)))
            e_temp.append(e)
            grad_e = (np.multiply((y - t[j]), X[j].T))
            w = w - eta[i]*grad_e
        e_all[i].append(np.mean(e_temp))
        if iter>0:
            if np.absolute(np.mean(e_temp)-e_all[i][iter-1]) < tol:
                break

plt.figure()
for i in range(len(e_all)):
    plt.plot(e_all[i])
plt.ylabel('Negative log likelihood')
plt.title('Training logistic regression')
plt.xlabel('Epoch')
plt.legend(eta)
plt.show()
# plt.savefig('plots-sgd/fig3-'+str(max_iter)+'.jpg',bbox_inches = 'tight')
