import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.stats as st
import csv
from IrisVector import IrisVector

#bayes rule
#decision
#classifier
#gradient
#nonlinearity


# returns a subplot of the iris data (since we'll be using it alot)
def getPlot(x_1, y_1, x_2, y_2, index):
	sub_plot = plt.subplot(index)
	petal_ve= plt.scatter(x_1, y_1 , marker = '*', color = 'g')
	petal_vi = plt.scatter(x_2, y_2, marker = 'x', color = 'b')
	plt.xlabel("petal length (cm)")
	plt.ylabel("petal width (cm)")
	axes = plt.gca()
	axes.set_xlim([2.5,7.5])
	axes.set_ylim([0.8, 2.6])
	plt.legend((petal_vi, petal_ve), ('Virginica', ' Versicolor'), loc = 'upper left')
	return sub_plot	

#Simple Decision Boundary function (1.b)
def simpleThreshold(data_point):
	decision_boundary = 4.85
	if (data_point.pl >= decision_boundary):
		return "virginica"
	else:
		return "versicolor"

#Simple Circle Decision Boundary function
def circleThreshold(center, radius, exercise1_results, pic_label, inner_class):
	fig2 = plt.figure(2)
	fig2.add_subplot(getPlot(exercise1_results[0], exercise1_results[1], exercise1_results[2], exercise1_results[3], 111))
	circle1 = plt.Circle((center[0], center[1]), radius, color = 'r', fill = False)
	ax = fig2.gca()
	ax.add_artist(circle1)
	plt.savefig('Exercise1.d.circle'+ str(pic_label) + '.png',bbox_inches='tight')
	plt.close('all')
	counter = 0;

	for vector in exercise1_results[4]:
		if (distance([vector.pl, vector.pw], center) <= radius) & (vector.name != inner_class):
			counter = counter + 1
		if (distance([vector.pl, vector.pw], center) > radius) & (vector.name == inner_class):
			counter = counter + 1
	print(str(counter) + " misclassifications for this circle decision boundary, centered at (" + str(center[0]) +"," + str(center[1]) 
		+ "), with radius of:" + str(radius))

def distance(vector_1, vector_2):
	return math.sqrt((vector_1[0] - vector_2[0])**2 + (vector_1[1] - vector_2[1])**2)
##############
#Exercise 1.a#
##############
#Pull data from the csv 
def exercise1():
	with open('irisdata.csv', newline='') as csvfile:
		datareader = csv.reader(csvfile, delimiter=' ',quotechar='|')
		counter = 1
		vector_list = []
		#Loop through each row of the csv
		for row in datareader:
			# we don't want any of the setosa irises
			if counter >= 52:
				joined_row = (', '.join(row))
				#create a new vector and append it to our list
				vector = IrisVector(joined_row)
				vector_list.append(vector)
			counter = counter + 1

	#initialize default np arrays
	petal_length_versicolor = np.array(range(1,51), dtype = float)
	petal_length_virginica = np.array(range(1,51), dtype = float)
	petal_width_versicolor = np.array(range(1,51), dtype = float)
	petal_width_virginica = np.array(range(1,51), dtype = float)
	versicolor_counter = 0;
	virginica_counter = 0;

	#Loop through every vector, adding them to their appropriate lists
	for vector in vector_list:
		if vector.name == "versicolor":
			petal_length_versicolor  = np.insert(petal_length_versicolor,versicolor_counter, vector.pl)
			petal_width_versicolor  = np.insert(petal_width_versicolor,versicolor_counter, vector.pw)
			versicolor_counter = versicolor_counter + 1
		else:
			petal_length_virginica  = np.insert(petal_length_virginica ,virginica_counter, vector.pl)
			petal_width_virginica  = np.insert(petal_width_virginica ,virginica_counter, vector.pw)
			virginica_counter = virginica_counter + 1

	#Trim the np arrays
	petal_length_versicolor = petal_length_versicolor[0:versicolor_counter]
	petal_width_versicolor = petal_width_versicolor[0:versicolor_counter]
	petal_length_virginica = petal_length_virginica[0:virginica_counter]
	petal_width_virginica = petal_width_virginica[0:virginica_counter]

	#Plot the points on the scatter plot
	fig1 = plt.figure(1)
	fig1.add_subplot(getPlot(petal_length_versicolor, petal_width_versicolor, petal_length_virginica, petal_width_virginica, 111))
	plt.savefig('Exercise1a.png',bbox_inches='tight')
	plt.close()

	##############
	#Exercise 1.b#
	##############
	#Add the vertical decision boundary
	fig2 = plt.figure(2)
	fig2.add_subplot(getPlot(petal_length_versicolor, petal_width_versicolor, petal_length_virginica, petal_width_virginica, 111))
	plt.axvline(x=4.85, color = 'r')
	plt.savefig('Exercise1b.png',bbox_inches='tight')
	plt.close()

	#Returns info needed to replot scatter later
	results = [petal_length_versicolor, petal_width_versicolor, petal_length_virginica, petal_width_virginica, vector_list]
	return results

def demonstrations():
	print("DEMONSTRATING EXERCISE 1.\n~~~~~~~~~~~~~~~~~~~~~~~~~")

	exercise1_results = exercise1()

	#Exercise 1c: testing the linear decision boundary:
	versicolor_vector = exercise1_results[4][0]
	virginica_vector = exercise1_results[4][50]
	print("EXERCISE 1C: Testing hand-drawn decision boundary:")
	print("\nExpected: " + versicolor_vector.name + "\nDecided: " + simpleThreshold(versicolor_vector))
	print("\nExpected: " + virginica_vector.name + "\nDecided: " + simpleThreshold(virginica_vector))
	
	#Exercise 1d: testing the circle decision boundary for three different centers:
	###############

	#4.5, 1.4, 0.5 radius
	circleThreshold([4.5,1.4],0.5, exercise1_results, 1, 'versicolor')
	circleThreshold([5.8,2.1],1, exercise1_results, 2, 'virginica')
	circleThreshold([4.2,1.3],0.8, exercise1_results, 3, 'versicolor')

	return exercise1_results

