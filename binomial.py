import numpy as np
import pandas as pd
from scipy import optimize

np.random.seed(1234)
Nfeval = 1

def callback(x):
	global Nfeval
	# x is the current parameter vector
	print('{}	{}'.format(Nfeval, x))
	Nfeval += 1

def binomial(x, args):
	# Calculate the negative log likelihood
	k = args
	if x <= 0:
		x = 1e-12
	if x >= 1:
		x = 1 - 1e-12
	nLL = -(np.sum(k)*np.log(x) + (len(k)-np.sum(k))*np.log(1-x))
	#print('nLL: {}'.format(nLL))
	return nLL

if __name__ == '__main__':

	# generate bernoulli distributed data with p = 0.2
	# we want to identify p using numeric MLE
	n = 1000
	sample_data = np.random.binomial(n=1, p=0.8, size=n)
	print('data: {}'.format(sample_data))

	# analytic form of p from algebraic MLE
	p = np.sum(sample_data)/n
	print('p: {}'.format(p))

	# initial guess for p
	x0 = 0.5
	nll = binomial(x0, sample_data)
	print(nll)

	# minimize the negative log-likelihood
	result = optimize.minimize(fun = binomial, 
		x0 = x0, 
		args = sample_data, 
		method = 'L-BFGS-B',
		# bounds = ((0, 1),),
		callback = callback,
		options = {'maxiter': 10000, 'disp': True})

	print('Solution: x = {}'.format(result.x))

	# plot the nLL
	x = np.arange(0, 1, 0.01)
	y = []
	for x0 in x:
		val = binomial(x0, sample_data)
		y.append(val)

	import matplotlib.pyplot as plt
	plt.plot(x, y, 'r-', label='nLL(x)')
	plt.vlines(result.x[0], 0, 2000, linestyle='--', label='x0')
	plt.ylim([0, 2000])
	plt.title('Negative Log-Likelihood')
	plt.xlabel('x')
	plt.ylabel('nLL(x)')
	plt.legend(loc='best')
	plt.show()

