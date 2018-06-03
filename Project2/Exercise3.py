import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.stats as st
import Exercise1 as ex1
import Exercise2 as ex2

def gradient_descent(ex1_results, data_set, w_t, epsilon, trial_num):
	previous_error = 10000000
	current_error = ex2.calculate_error(data_set, w_t)
	w_t_new = w_t
	w_ts = []
	errors = []
	iterations = []
	counter = 1
	
	while((previous_error - current_error) > 0.5):

		
		w_t_new = ex2.update_weights(data_set, w_t_new, epsilon)
		w_ts.append(w_t_new)
		print(w_t_new)
	
		previous_error = current_error
		current_error = ex2.calculate_error(data_set, w_t_new)
		errors.append(current_error)

		iterations.append(counter)
		
		update_plot(ex1_results, iterations,errors, w_ts, counter-1, -1, trial_num)
	
		counter = counter + 1

	start = 0
	end = len(iterations) - 1
	middle = int(end/2)
	update_plot(ex1_results, iterations,errors, w_ts, start, 1, trial_num)
	update_plot(ex1_results, iterations,errors, w_ts, middle, 1, trial_num)
	update_plot(ex1_results, iterations,errors, w_ts, end, 1, trial_num)
	return w_t_new

def update_plot(ex1_results, iterations, errors, w_ts, index, save_picture, trial_num):
	fig1 = plt.figure(1)
	fig1.add_subplot(ex1.getPlot(ex1_results[0], ex1_results[1], ex1_results[2], ex1_results[3], 211))
	
	x_1 = np.linspace(-200.,200., 1000)
	d = ex2.decisionBoundary(w_ts[index][1], w_ts[index][2], x_1, w_ts[index][0])
	plt.plot(x_1, d, 'r')

	errorgraph = fig1.add_subplot(2,1,2)
	errorgraph.plot(iterations[0:index+1], errors[0:index+1], '-o')
	errorgraph.set_xlabel("iterations")
	errorgraph.set_ylabel("mean-squared error")	
	if(save_picture == 1):
		plt.savefig(str(trial_num) + 'Exercise3.' + str(index) + '.png',bbox_inches='tight')
	else:
		print("exit picture to continue")
		plt.show()
	plt.close()

def demonstrations(ex1_results):
	print("\n\nDEMONSTRATING EXERCISE 3.\n~~~~~~~~~~~~~~~~~~~~~~~~~")
	data_set = ex1_results[4]

	#gradient_descent(ex1_results, data_set, [-4,1,0.1], 0.1, 0)
	w = [20,1,-1]
	while not((float(w[2] + w[0])/float(w[1]) < -3) & (float(w[2] + w[0])/float(w[1]) > -7)):
		w = [np.random.randint(-10,-2), np.random.randint(1,10), np.random.randint(1,10)]
	gradient_descent(ex1_results, data_set, w, 0.1, 1)


