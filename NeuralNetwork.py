class NeuralNetwork():

	def __init__(self, X, Y, layers_dims, learning_rate, num_iterations, print_cost, initialization):
		self.X = X
		self.Y = Y
		self.layers_dims = layers_dims
		self.learning_rate = learning_rate
		self.num_iterations = num_iterations
		self.print_cost = print_cost
		self.initialization = initialization
		self.params = {}


	def summary(self):
		print('Train Data shape:', self.X.shape)
		print('Test Data shape:', self.Y.shape)
		print('NN Architecture:', self.layers_dims)
		print('Learning Rate \u03B1:', self.learning_rate)
		print('Number of Iterations:', self.num_iterations)
		print('Is it printing cost values?', str(self.print_cost))
		print('Paramaters:', self.params)


	def fit(self):
		self.costs = []
		self.params = initialize_parameters(self.layers_dims, self.initialization)
		for i in range(0, self.num_iterations):
			# Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
			self.AL, self.caches = forward_propagation(self.X, self.params)
			# Compute cost. 
			self.cost = compute_cost(self.AL, self.Y)
			
			# Backward propagation.
			self.grads = backward_propagation(self.AL, self.Y, self.caches)

			# Update parameters.
			self.params = update_parameters(self.params, self.grads, self.learning_rate)
					
			# Print the cost every 100 training example
			if self.print_cost and i % 100 == 0:
				print ("Cost after iteration %i: %f" %(i, self.cost))
			if self.print_cost and i % 100 == 0:
				self.costs.append(self.cost)
				
		# plot the cost
		plt.plot(np.squeeze(self.costs))
		plt.ylabel('cost')
		plt.xlabel('iterations (per hundreds)')
		plt.title("Learning rate =" + str(self.learning_rate))
		plt.show()
		
		return self.params


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

def sigmoid(Z):
	"""
	Implements the sigmoid activation in numpy
	
	Arguments:
	Z -- numpy array of any shape
	
	Returns:
	A -- output of sigmoid(z), same shape as Z
	cache -- returns Z as well, useful during backpropagation
	"""
	A = 1/(1+np.exp(-Z))
	cache = Z

	return A, cache


def relu(Z):
	"""
	Implement the RELU function.

	Arguments:
	Z -- Output of the linear layer, of any shape

	Returns:
	A -- Post-activation parameter, of the same shape as Z
	cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
	"""
	A = np.maximum(0,Z)
	assert(A.shape == Z.shape)
	cache = Z 

	return A, cache


def relu_backward(dA, cache):
	"""
	Implement the backward propagation for a single RELU unit.

	Arguments:
	dA -- post-activation gradient, of any shape
	cache -- 'Z' where we store for computing backward propagation efficiently

	Returns:
	dZ -- Gradient of the cost with respect to Z
	"""
	Z = cache
	dZ = np.array(dA, copy=True) # just converting dz to a correct object.
	# When z <= 0, you should set dz to 0 as well. 
	dZ[Z <= 0] = 0
	assert (dZ.shape == Z.shape)
	
	return dZ


def sigmoid_backward(dA, cache):
	"""
	Implement the backward propagation for a single SIGMOID unit.

	Arguments:
	dA -- post-activation gradient, of any shape
	cache -- 'Z' where we store for computing backward propagation efficiently

	Returns:
	dZ -- Gradient of the cost with respect to Z
	"""
	Z = cache
	s = 1/(1+np.exp(-Z))
	dZ = dA * s * (1-s)
	assert (dZ.shape == Z.shape)

	return dZ


def initialize_parameters(layer_dims, initialization):
	"""
	Weights initialization
	Arguments:
	layer_dims -- python array (list) containing the dimensions of each layer in our network
	initialization -- Gaussian variance (see notes): 
	             random, he (He et. al), xavier (Xavier et. al), ybengio (Yoshua Bengio et al)

	Returns:
	parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
					Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
					bl -- bias vector of shape (layer_dims[l], 1)
	"""
	parameters = {}
	L = len(layer_dims)            # number of layers in the network

	if initialization == 'he':
		for l in range(1, L): 
			parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2/layers_dims[l-1])
			parameters['b' + str(l)] = np.zeros((layer_dims[l],1))
			assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
			assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
	elif initialization == 'xavier':
		for l in range(1, L): 
			parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(1/layers_dims[l-1])
			parameters['b' + str(l)] = np.zeros((layer_dims[l],1))
			assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
			assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
	elif initialization == 'ybengio':
		for l in range(1, L): 
			parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2/(layers_dims[l-1]+layers_dims[l]))
			parameters['b' + str(l)] = np.zeros((layer_dims[l],1))
			assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
			assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
	elif initialization == 'random':
		for l in range(1, L): 
			parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
			parameters['b' + str(l)] = np.zeros((layer_dims[l],1))
			assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
			assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
	elif initialization == 'zeros':
		for l in range(1, L): 
			parameters['W' + str(l)] = np.zeros((layer_dims[l], layer_dims[l-1]))
			parameters['b' + str(l)] = np.zeros((layer_dims[l],1))
			assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
			assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

	return parameters

def linear_forward(A, W, b):
	"""
	Implement the linear part of a layer's forward propagation.

	Arguments:
	A -- activations from previous layer (or input data): (size of previous layer, number of examples)
	W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
	b -- bias vector, numpy array of shape (size of the current layer, 1)

	Returns:
	Z -- the input of the activation function, also called pre-activation parameter 
	cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
	"""
	Z = np.dot(W,A) + b
	assert(Z.shape == (W.shape[0], A.shape[1]))
	cache = (A, W, b)	

	return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
	"""
	Implement the forward propagation for the LINEAR->ACTIVATION layer

	Arguments:
	A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
	W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
	b -- bias vector, numpy array of shape (size of the current layer, 1)
	activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

	Returns:
	A -- the output of the activation function, also called the post-activation value 
	cache -- a python tuple containing "linear_cache" and "activation_cache";
			 stored for computing the backward pass efficiently
	"""
	if activation == "sigmoid":
		# Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
		Z, linear_cache = linear_forward(A_prev, W, b)
		A, activation_cache = sigmoid(Z)
	elif activation == "relu":
		# Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
		Z, linear_cache = linear_forward(A_prev, W, b)
		A, activation_cache = relu(Z)
	assert (A.shape == (W.shape[0], A_prev.shape[1]))
	cache = (linear_cache, activation_cache)

	return A, cache


def forward_propagation(X, parameters):
	"""
	Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
	
	Arguments:
	X -- data, numpy array of shape (input size, number of examples)
	parameters -- output of initialize_parameters()
	
	Returns:
	AL -- last post-activation value
	caches -- list of caches containing:
				every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
	"""
	caches = []
	A = X
	L = len(parameters) // 2                  # number of layers in the neural network
	# Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
	for l in range(1, L):
		A_prev = A
		A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], 'relu')
		caches.append(cache)
	
	# Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
	AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], 'sigmoid')
	caches.append(cache)
	assert(AL.shape == (1,X.shape[1]))

	return AL, caches


def compute_cost(AL, Y):
	"""
	Implement the cost function defined by equation (7).

	Arguments:
	AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
	Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

	Returns:
	cost -- cross-entropy cost
	"""
	m = Y.shape[1]
	# Compute loss from aL and y.
	cost = (-1/m)*np.sum(Y*np.log(AL)+(1-Y)*np.log(1-AL), axis=1)
	cost = np.squeeze(cost)      
	# To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
	assert(cost.shape == ())
	
	return cost


def linear_backward(dZ, cache):
	"""
	Implement the linear portion of backward propagation for a single layer (layer l)

	Arguments:
	dZ -- Gradient of the cost with respect to the linear output (of current layer l)
	cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

	Returns:
	dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
	dW -- Gradient of the cost with respect to W (current layer l), same shape as W
	db -- Gradient of the cost with respect to b (current layer l), same shape as b
	"""
	A_prev, W, b = cache 
	m = A_prev.shape[1]
	dW = (1/m)*np.dot(dZ,A_prev.T)
	db = (1/m)*np.sum(dZ, axis=1, keepdims=True)
	dA_prev = np.dot(W.T,dZ)
	assert (dA_prev.shape == A_prev.shape)
	assert (dW.shape == W.shape)
	assert (db.shape == b.shape)
	
	return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
	"""
	Implement the backward propagation for the LINEAR->ACTIVATION layer.
	
	Arguments:
	dA -- post-activation gradient for current layer l 
	cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
	activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
	
	Returns:
	dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
	dW -- Gradient of the cost with respect to W (current layer l), same shape as W
	db -- Gradient of the cost with respect to b (current layer l), same shape as b
	"""
	linear_cache, activation_cache = cache
	if activation == "relu":
		### START CODE HERE ### (≈ 2 lines of code)
		dZ = relu_backward(dA, activation_cache)
		dA_prev, dW, db = linear_backward(dZ, linear_cache)
		### END CODE HERE ###
	elif activation == "sigmoid":
		### START CODE HERE ### (≈ 2 lines of code)
		dZ = sigmoid_backward(dA, activation_cache)
		dA_prev, dW, db = linear_backward(dZ, linear_cache)
		### END CODE HERE ###
	return dA_prev, dW, db


def backward_propagation(AL, Y, caches):
	"""
	Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
	
	Arguments:
	AL -- probability vector, output of the forward propagation (forward_propagation())
	Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
	caches -- list of caches containing:
				every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
				the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
	
	Returns:
	grads -- A dictionary with the gradients
			 grads["dA" + str(l)] = ... 
			 grads["dW" + str(l)] = ...
			 grads["db" + str(l)] = ... 
	"""
	grads = {}
	L = len(caches) # the number of layers
	m = AL.shape[1]
	Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
	# Initializing the backpropagation
	dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))  
	# Lth layer (SIGMOID -> LINEAR) gradients. 
	# Inputs: "dAL, current_cache". 
	# Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
	current_cache = caches[L-1]
	grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, 'sigmoid')
	
	# Loop from l=L-2 to l=0
	for l in reversed(range(L-1)):
		# lth layer: (RELU -> LINEAR) gradients.
		# Inputs: "grads["dA" + str(l + 1)], current_cache". 
		# Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
		current_cache = caches[l]
		dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+1)], current_cache, 'relu')
		grads["dA" + str(l)] = dA_prev_temp
		grads["dW" + str(l + 1)] = dW_temp
		grads["db" + str(l + 1)] = db_temp

	return grads


def update_parameters(parameters, grads, learning_rate):
	"""
	Update parameters using gradient descent
	
	Arguments:
	parameters -- python dictionary containing your parameters 
	grads -- python dictionary containing your gradients, output of backward_propagation
	
	Returns:
	parameters -- python dictionary containing your updated parameters 
				  parameters["W" + str(l)] = ... 
				  parameters["b" + str(l)] = ...
	"""
	L = len(parameters) // 2 # number of layers in the neural network
	# Update rule for each parameter:
	for l in range(L):
		parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*grads["dW" + str(l+1)]
		parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*grads["db" + str(l+1)]

	return parameters

def predict(X, y, parameters):
    """
    This function is used to predict the results of a  L-layer neural network.
    
    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    
    Returns:
    p -- predictions for the given dataset X
    """
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))
    # Forward propagation
    probas, caches = forward_propagation(X, parameters)
    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    print("Accuracy: "  + str(np.sum((p == y)/m)))
        
    return p

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################

import h5py
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
	np.random.seed(1)
	train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
	train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   
	# The "-1" makes reshape flatten the remaining dimensions
	test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
	# Standardize data to have feature values between 0 and 1.
	train_x = train_x_flatten/255.
	test_x = test_x_flatten/255.

	# HYPERPARAMETERS
	layers_dims = [train_x.shape[0], 20, 7, 5, 1]
	num_iterations = 3000
	learning_rate = 0.0075
	print_cost = True	
	initialization = 'zeros'
	nn = NeuralNetwork(train_x, train_y, layers_dims, learning_rate, num_iterations, print_cost, initialization)
	nn_params = nn.fit()
	nn.summary()
	predictions = predict(test_x, test_y, nn_params)
	