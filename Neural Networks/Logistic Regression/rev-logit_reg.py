import numpy as np
import matplotlib.pyplot as plt
import h5py
from lr_utils import load_dataset

X_train_orig, Y, X_test_orig, Y_test, classes = load_dataset()

X = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test = X_test_orig.reshape(X_test_orig.shape[0], -1).T

X = X/255.
X_test = X_test/255.

def sigmoid(z):
	return 1 / (1 + np.exp(-z))


def initialize(dim):
	w = np.zeros(shape=(dim,1))
	b = 0
	return w, b


def propagate(w, b, X, Y):
	m = X.shape[1]
	A = sigmoid(np.dot(w.T, X) + b)
	cost = (-1/m)*(np.sum(Y*np.log(A)+(1-Y)*np.log(1-A)))

	dw = (1/m)*np.dot(X, (A-Y).T)
	db = (1/m)*np.sum(A-Y)

	grads = {'dw':dw, 'db':db}

	return grads, cost


def optimize(w, b, X, Y, N_ITER, LEARNING_RATE):
	costs = []
	for i in range(N_ITER):
		grads, cost = propagate(w, b, X, Y)

		dw = grads['dw']
		db = grads['db']

		# GD
		w = w - LEARNING_RATE*dw
		b = b - LEARNING_RATE*db
		if i % 100 == 0:
			costs.append(cost)

	params = {'w':w, 'b':db}
	grads = {'dw':w, 'db':db}

	return params, grads, costs

def predict(w, b, X):
	A = sigmoid(np.dot(w.T, X) + b)
	predictions = [1 if A[0][x] > 0.5 else 0 for x in range(A.shape[1])]
	return predictions

def model(X, Y, X_test, Y_test, N_ITER = 2000, LEARNING_RATE = 0.5):
	dim = X.shape[0]
	w, b = initialize(dim)
	parameters, gradients, costs = optimize(w, b, X, Y, N_ITER, LEARNING_RATE)
	w = parameters['w']
	b = parameters['b']

	Y_pred_train = predict(w, b, X)
	Y_pred_test = predict(w, b, X_test)

	print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_pred_train - Y)) * 100))
	print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_pred_test - Y_test)) * 100))

	d = {"costs": costs,
	"Y_pred": Y_pred_test,
	"Y_pred" : Y_pred_train,
	"w" : w,
	"b" : b,
	"LEARNING_RATE" : LEARNING_RATE,
	"N_ITER": N_ITER}

	return d

d = model(X, Y, X_test, Y_test, N_ITER = 2000, LEARNING_RATE = 0.005)
print(d)