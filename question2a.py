import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from common import *

print("Question 2a")
print("Laplace Approximation\n")

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

delta = 0.05
wmax  = 5
wi    = np.arange(-wmax, wmax+ delta, delta)
wi1   = np.arange(-2.8, -2.6+delta, delta)
wi2   = np.arange(-1.7, -1.5+delta, delta)
wi3   = np.arange(-2.0, -1.8+delta, delta)
wi4   = np.arange(-0.2, -0.1+delta, delta)
wi5   = np.arange( 2.7,  2.9+delta, delta)
w_mesh= np.array(np.meshgrid(wi1, wi2, wi3, wi4, wi5)).T.reshape(-1,5)

max_log_likelihood = -100000
w_map = np.zeros((5, ))

# First get the mode
for i in range(w_mesh.shape[0]):
	dot_product = np.matmul(train_data,w_mesh[i])
	pred = sigmoid(dot_product)
	likeli = log_likelihood(phi= train_data, pred= pred, t= train_label[:, np.newaxis], dot_product= dot_product, weight= w_mesh[i])
	if likeli > max_log_likelihood:
		max_log_likelihood = likeli
		w_map = w_mesh[i]

dot_product = np.matmul(train_data,w_map)
pred = sigmoid(dot_product)

hessian  = hessian (phi= train_data, pred= pred[:, np.newaxis], t= train_label[:, np.newaxis], dot_product= dot_product)

print("W_MAP = ")
print(w_map)
print("S = ")
print(hessian)

dot_product = np.matmul(test_data, w_map)
w_map_test = sigmoid(dot_product)

# Calculate on test data
if w_map_test.ndim == 2:
	w_map_test = w_map_test[:,0]
w_map_test[w_map_test >= 0.5] = 1.0
w_map_test[w_map_test <  0.5] = 0.0
acc = np.sum(w_map_test == test_label)*100.0/w_map_test.shape[0]
print("Test Accuracy: {:.2f}".format(acc))

pred_like = np.zeros((test_data.shape[0],))

for i in range(test_data.shape[0]):
	test_pt = test_data[i]
	m = np.sqrt(2 * np.matmul(np.matmul(test_pt[np.newaxis,:], hessian), test_pt[:, np.newaxis]))
	b = np.sum (np.multiply(w_map, test_data[i]))
	likelihood_1 = gass_hermite_quad(inner, degree= 100, m= m, b= b)/np.sqrt(np.pi)
	pred_like[i] = likelihood_1**test_label[i] * (1-likelihood_1)**(1-test_label[i])
pred_likelihood = np.mean(pred_like)
print("Prediction Likelihood Average: {:.2f}".format(pred_likelihood))

print("=====================================================================")