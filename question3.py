import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from common import *

train = pd.read_csv("data/faithful/faithful.txt", delimiter= " ", header=None)

train.iloc[:, 0] = (train.iloc[:, 0] - train.iloc[:, 0].mean()) / (train.iloc[:, 0].max() - train.iloc[:, 0].min())
train.iloc[:, 1] = (train.iloc[:, 1] - train.iloc[:, 1].mean()) / (train.iloc[:, 1].max() - train.iloc[:, 1].min())

def E_Update():
	pass

def M_Update():
	pass

cluster_means = np.array([[-1, 1], [1, -1]])