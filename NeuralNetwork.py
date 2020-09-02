class NeuralNetwork():

	def __init__(self, X, Y, layers_dims, learning_rate, num_iterations=0, print_cost=0, params=0):
		self.X = X
		self.Y = Y
		self.layers_dims = layers_dims
		self.num_iterations = num_iterations
		self.params = params

	def get_params(self):
		return self.params

	def get_layers(self):
		return self.layers_dims

	def set_var(self, arg):
		self.params = arg
		pass

	def initialize_parameters(self, layer_dims):
		"""
		Arguments:
		layer_dims -- python array (list) containing the dimensions of each layer in our network

		Returns:
		parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
		                Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
		                bl -- bias vector of shape (layer_dims[l], 1)
		"""
		parameters = {}
		L = len(layer_dims)            # number of layers in the network

		for l in range(1, L): 
		    parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
		    parameters['b' + str(l)] = np.zeros((layer_dims[l],1))
		    assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
		    assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

		return parameters


def load_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################

import h5py
import numpy as np

if __name__ == '__main__':

	train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
	train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   
	# The "-1" makes reshape flatten the remaining dimensions
	test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
	# Standardize data to have feature values between 0 and 1.
	train_x = train_x_flatten/255.
	test_x = test_x_flatten/255.

	# HYPERPARAMETERS
	layers_dims = [3, 4, 5, 2, 1]
	num_iterations = 2500
	learning_rate = 0.0075
	print_cost = True
	params = 1000	
	nn = NeuralNetwork(params, layers_dims)
	