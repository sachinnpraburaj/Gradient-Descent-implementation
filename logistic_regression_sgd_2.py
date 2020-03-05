#!/usr/bin/env python

# Run logistic regression training.

import numpy as np
import scipy.special as sps
import matplotlib.pyplot as plt
import data_utils as a2


# Maximum number of iterations.  Continue until this limit, or when error change is below tol.
max_iter = 500
tol = 0.00001

# Step size for gradient descent.
eta = [0.5, 0.3, 0.1, 0.05, 0.01]


# Load data.
data = np.genfromtxt('data.txt')

# Data matrix, with column of ones at end.
X = data[:,0:3]
# Target values, 0 for class 1, 1 for class 2.
t = data[:,3]
# For plotting data
class1 = np.where(t==0)
X1 = X[class1]
class2 = np.where(t==1)
X2 = X[class2]




# Error values over all iterations.
e_all = [[] for x in range(len(eta))]


def random_sample(X,t,samplesize):
    index = np.random.choice(X.shape[0],size=samplesize,replace=False)
    X_sub = X[index,:]
    t_sub = []
    for j in index:
        t_sub.append(t[j])
    t_sub = np.array(t_sub)
    return (X_sub,t_sub)

for i in range(len(eta)):
    w = np.array([0.1, 0, 0])
    for iter in range (0,max_iter):
        # Compute output using current w on subset data X_sub.
        (X_sub,t_sub) = random_sample(X,t,samplesize=128)

        y = sps.expit(np.dot(X_sub,w))

        # e is the error, negative log-likelihood (Eqn 4.90)
        e = -np.mean(np.multiply(t_sub,np.log(y+0.00001)) + np.multiply((1-t_sub),np.log(1-y+0.00001)))

        # Add this error to the end of error vector.
        e_all[i].append(e)


        # Gradient of the error, using Eqn 4.91
        grad_e = np.mean(np.multiply((y - t_sub), X_sub.T), axis=1)

        w = w - eta[i]*grad_e

        if iter>0:
            if np.absolute(e-e_all[i][iter-1]) < tol:
                print ('epoch {0:d}, negative log-likelihood {1:.4f}, w={2}'.format(iter, e, w.T))
                break

        if iter==499:
            print ('epoch {0:d}, negative log-likelihood {1:.4f}, w={2}'.format(iter, e, w.T))

# Plot error over iterations
plt.figure()
for i in range(len(e_all)):
    plt.plot(e_all[i])
plt.ylabel('Negative log likelihood')
plt.title('Training logistic regression')
plt.xlabel('Epoch')
plt.legend(eta)
plt.show()
#plt.savefig('plots-sgd/sgd2_fig3-'+str(max_iter)+'.jpg',bbox_inches = 'tight')
