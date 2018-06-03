import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.stats as st
import Exercise1 as ex1


#Functional representation of our decision boundary used in 2&3
decisionBoundary = lambda w_1, w_2, x_1, b:-((w_1*x_1 + b)/w_2)

#Exercise 2.a: Calculates the Mean-squared error for the set
def calculate_error(data_set, w_t):
	error = 0
	for vector in data_set:
		if (vector.name == "versicolor"):
			c = -1
		else:
			c = 1
		x = np.array([1, vector.pl,vector.pw])
		error = np.add(error, np.power(np.dot(w_t,x) - c,2))

	return error

#Exercise 2.e: Calculates the gradient of the mean-squared error
def calculate_gradient(data_set, w_t):
	gradient = [0,0,0]
	for vector in data_set:
		if (vector.name == "versicolor"):
			c = -1
		else:
			c = 1
		x = np.array([1, vector.pl, vector.pw])
		#((w_t*x - c)*x)
		gradient = np.add(gradient, np.multiply((np.dot(w_t,x)-c),x))

	return gradient

def update_weights(data_set, w_t, epsilon):
	step = epsilon/1000
	gradient = calculate_gradient(data_set, w_t)
	w_t_new = np.subtract(w_t, np.multiply(gradient, step))
	return w_t_new

def demonstrations(ex1_results):	
	print("\n\nDEMONSTRATING EXERCISE 2.\n~~~~~~~~~~~~~~~~~~~~~~~~~")
	data_set = ex1_results[4]


	small_error = calculate_error(data_set, [-5,1,0.1])
	fig1 = plt.figure(1)
	fig1.add_subplot(ex1.getPlot(ex1_results[0], ex1_results[1], ex1_results[2], ex1_results[3], 211))
	x_1 = np.linspace(0.,7., 1000)
	d = decisionBoundary(1, 0.1, x_1, -5)
	plt.plot(x_1, d, 'y')
	plt.savefig('Exercise2bsmall.png', bbox_inches='tight')

	large_error = calculate_error(data_set, [-12,1,8])
	fig2 = plt.figure(2)
	fig2.add_subplot(ex1.getPlot(ex1_results[0], ex1_results[1], ex1_results[2], ex1_results[3], 211))
	x_1 = np.linspace(0.,7., 1000)
	d = decisionBoundary(1, 8, x_1, -12)
	plt.plot(x_1, d, 'y')
	plt.savefig('Exercise2blarge.png', bbox_inches='tight')

	print("EXERCISE 2B: computing errors for different decision boundaries:\n")
	print("Small error using w = [-5,1,0]: " + str(small_error))
	print("Large error using w = [-1,1,0]: " + str(large_error))


	print("\nEXERCISE 2E: original weights: [-4,1,0.1]")
	w = [-4, 1, 0.1]
	fig3 = plt.figure(3)
	fig3.add_subplot(ex1.getPlot(ex1_results[0], ex1_results[1], ex1_results[2], ex1_results[3], 211))
	x_1 = np.linspace(0.,7., 1000)
	d = decisionBoundary(w[1], w[2], x_1, w[0])
	plt.plot(x_1, d, 'y')

	print("Updating weights, with a step size of epsilon = 0.1...")
	w = update_weights(data_set, w, 0.1)
	print("New weights: [" + str(w[0]) + "," + str(w[1]) + "," + str(w[2]) + "]")
	fig3.add_subplot(ex1.getPlot(ex1_results[0], ex1_results[1], ex1_results[2], ex1_results[3], 212))
	x_1 = np.linspace(0.,7., 1000)
	d = decisionBoundary(w[1], w[2], x_1, w[0])
	plt.plot(x_1, d, 'r')
	plt.savefig('Exercise2e.png', bbox_inches='tight')
	plt.close('all')



