import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from common import *
from scipy.stats import multivariate_normal

train = pd.read_csv("data/faithful/faithful.txt", delimiter= " ", header=None)

train.iloc[:, 0] = (train.iloc[:, 0] - train.iloc[:, 0].mean()) / (train.iloc[:, 0].max() - train.iloc[:, 0].min())
train.iloc[:, 1] = (train.iloc[:, 1] - train.iloc[:, 1].mean()) / (train.iloc[:, 1].max() - train.iloc[:, 1].min())

def E_Update(X, num_clusters, cluster_means, cov, alpha):
	weights = np.zeros((X.shape[0], num_clusters))
	for i in range(num_clusters):
		weights[:,i] = alpha[i] * multivariate_normal.pdf(X, mean=cluster_means[i], cov=cov[i])
	return np.divide(weights, np.sum(weights, axis=1)[:,np.newaxis])

def M_Update(X, num_clusters, weights):
	num_rows = X.shape[0]
	num_features = X.shape[1]
	total = np.sum(weights, axis=0)
	alpha = total/num_rows
	cluster_means = np.zeros((num_clusters, num_features))
	cov = np.zeros((num_clusters, num_features, num_features))

	for i in range(num_clusters):
		reshaped_means = np.repeat(weights[:,i], num_features).reshape(num_rows, num_features)
		reshaped_cov = np.repeat(weights[:,i], num_features*num_features).reshape(num_rows, num_features, num_features)
		cluster_means[i] = np.sum(np.multiply(reshaped_means, X), axis=0) / total[i]
		N2 = X - cluster_means[i]
		outer   = np.matmul(N2[:,:,np.newaxis], N2[:,np.newaxis,:])
		cov[i] = np.sum(np.multiply(reshaped_cov, outer), axis=0) / total[i]
	return alpha, cluster_means, cov

iterations = 100
cluster_means = np.array([[-1, 1], [1, -1]])
num_clusters  = 2
cov = np.zeros((num_clusters, 2, 2))
cov[0] = 0.1*np.eye(num_clusters)
cov[1] = 0.1*np.eye(num_clusters)
alpha = (1/num_clusters) * np.ones((num_clusters,))

draws = [1,2,5,100]
train = train.to_numpy()
for i in range(iterations+1):
	weights = E_Update(train, num_clusters, cluster_means, cov, alpha)
	alpha, cluster_means, cov = M_Update(train, num_clusters, weights)

	if i in draws:
		x = weights[:,0] >= weights[:,1]
		y = weights[:,0] < weights[:,1]
		plt.scatter(train[x, 0], train[x, 1])
		plt.scatter(train[y, 0], train[y, 1])
		plt.scatter(cluster_means[:,0], cluster_means[:,1])
		plt.show()

cluster_means = np.array([[-1,-1],[1,1]])
cov = np.zeros((num_clusters, 2,2))
cov[0] = 0.5 * np.eye(num_clusters)
cov[1] = 0.5 * np.eye(num_clusters)
alpha = (1/num_clusters) * np.ones((num_clusters,))

for i in range(iterations+1):
	weights = E_Update(train, num_clusters, cluster_means, cov, alpha)
	alpha, cluster_means, cov = M_Update(train, num_clusters, weights)

	if i in draws:
		x = weights[:,0] >= weights[:,1]
		y = weights[:,0] < weights[:,1]
		plt.scatter(train[x, 0], train[x, 1])
		plt.scatter(train[y, 0], train[y, 1])
		plt.scatter(cluster_means[:,0], cluster_means[:,1])
		plt.show()