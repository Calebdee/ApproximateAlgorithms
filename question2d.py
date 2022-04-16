import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from common import *

train = np.genfromtxt("data/bank-note/train.csv", delimiter= ",")
test  = np.genfromtxt("data/bank-note/test.csv", delimiter= ",")
train_label = train[:,-1]
test_label  = test [:,-1]
train_data  = np.hstack((train[:,0:-1], np.ones((train.shape[0],1))))
test_data   = np.hstack((test [:,0:-1], np.ones((test.shape[0], 1))))

def inner(x=[], m=10, b=3):
    x = np.array([x])
    x = m*x + b
    out = np.zeros(x.shape)
    ind1 = (x >= 0)
    ind2 = (x  < 0)
    out[ind1] = 1 / (1 + np.exp(-x[ind1]))
    out[ind2] = np.divide(np.exp(x[ind2]), (1 + np.exp(x[ind2])))

    if type(x) != np.ndarray:
        return out[0]
    else:
        return out

print("Question 2d")
print("Factorized Variational Logistic Regression\n")

pred_like = np.zeros((test_data.shape[0],))
pred_likelihood = -100
dim = train_data.shape[1]

xi = -np.ones ((train_data.shape[0],))
mN =   np.zeros((dim, ))
mN_new = np.zeros((dim, ))
SN = np.zeros((dim, dim))

update_means_inplace = True
if update_means_inplace:
    print("Updating the means inplace")
else:
    print("Not updating the means inplace")

for i in range(100):
    lambdaa = np.multiply(1/(2*(xi + 1e-6)), (sigmoid(xi) - 0.5))

    phi_phi_transpose = np.matmul(train_data[:,:,np.newaxis], train_data[:,np.newaxis,:]) # N x 5 x 5
    phi_phi_transpose = np.multiply(lambdaa[:, np.newaxis, np.newaxis], phi_phi_transpose) # N x 5 x 5
    mean_term_1_all   = np.sum(np.multiply( (train_label[:, np.newaxis]-0.5), train_data), axis= 0) # 5
    
    # Do the Expectation step in multiple steps
    # Update one variable at a time.
    for j in range(dim):
        mean_term_1 = mean_term_1_all[j]
        all_except_one_index = np.delete(np.arange(dim), j)
        b = mN[np.newaxis, :]
        inner_sum         = np.sum( np.multiply(b[:,all_except_one_index], train_data[:,all_except_one_index]), axis=1) # N
        mean_term_2       = -2.0 * np.sum( np.multiply( np.multiply(lambdaa, train_data[:,j]), inner_sum), axis=0) # 1

        SN[j,j] = 1.0/( 1 + 2*np.sum(phi_phi_transpose[:,j,j], axis= 0))
        if update_means_inplace:
            mN[j]   = SN[j,j] * (mean_term_1 + mean_term_2)
        else:
            mN_new[j]   = SN[j,j] * (mean_term_1 + mean_term_2)

    if not update_means_inplace:
        mN = mN_new

    # Do the maximisation step
    temp = SN + np.matmul(mN[:, np.newaxis], mN[np.newaxis, :])
    xi = np.matmul( np.matmul( train_data[:, np.newaxis, :],  temp[np.newaxis, :, :]), train_data[:, :, np.newaxis]).flatten()
    xi = np.sqrt(xi)

    # Calculate on test data
    dot_product = np.matmul(test_data,mN)
    pred_test = sigmoid(dot_product)

    if pred_test.ndim == 2:
    	pred_test = pred_test[:,0]
    pred_test[pred_test >= 0.5] = 1.0
    pred_test[pred_test <  0.5] = 0.0
    acc = np.sum(pred_test == test_label)*100.0/pred_test.shape[0]
    for i in range(test_data.shape[0]):
    	test_pt = test_data[i]
    	m = np.sqrt(2 * np.matmul(np.matmul(test_pt[np.newaxis,:], SN), test_pt[:, np.newaxis]))
    	b = np.sum (np.multiply(mN, test_data[i]))
    	likelihood_1 = gass_hermite_quad(inner, degree= 100, m= m, b= b)/np.sqrt(np.pi)
    	pred_like[i] = likelihood_1**test_label[i] * (1-likelihood_1)**(1-test_label[i])
    pred_likelihood_new = np.mean(pred_like)

    if np.abs(pred_likelihood_new - pred_likelihood) < 1e-3:
        break
    else:
        pred_likelihood = pred_likelihood_new

print("\nMean = ")
print(mN)
print("S = ")
print(SN)
