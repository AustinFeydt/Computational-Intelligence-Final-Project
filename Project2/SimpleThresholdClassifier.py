#Functional representation of our decision boundary used in 2&3
decisionBoundary = lambda w_1, w_2, x_1, b:-((w_1*x_1 + b)/w_2)



results = exercise1()
fig3 = plt.figure(3)
fig3.add_subplot(getPlot(results[0], results[1], results[2], results[3], 111))

vector_list = results[4]

w = [-4.85,1,0]

for counter in range(0,1):
	t = calculate_gradient(vector_list,w)
	t = np.multiply(t, 0.0001)
	w = np.subtract(w, t)
	z = calculate_error(vector_list, w)
	
x_1 = np.linspace(0.,7., 1000)
d = decisionBoundary(w[1], w[2], x_1, w[0])
plt.plot(x_1, d, 'y')
plt.savefig('Exercise1e.png', bbox_inches='tight')
plt.close()

fig4 = plt.figure(4)
fig4.add_subplot(getPlot(results[0], results[1], results[2], results[3], 111))

w = [-7,3,1]
for counter in range(0,1):
	t = calculate_gradient(vector_list,w)
	t = np.multiply(t, 0.0001)
	w = np.subtract(w, t)
	z = calculate_error(vector_list, w)
d = decisionBoundary(w[1], w[2], x_1, w[0])


plt.plot(x_1, d, 'y')
plt.savefig('Test', bbox_inches='tight')

plt.close()


