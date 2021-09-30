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

def poisson(x, args):
	# Calculate the negative log likelihood
	k = args
	nLL = -(np.sum(k)*np.log(x) - len(k)*x)
	#print('nLL: {}'.format(nLL))
	return nLL

if __name__ == '__main__':

	# generate poisson distributed data with \lambda = 3
	# we want to identify \lambda using numeric MLE
	n = 1000
	sample_data = np.random.poisson(lam=3, size=n)
	print('data: {}'.format(sample_data))

	# analytic form of \lambda from algebraic MLE
	# \lambda = \sum(k_i)/n
	lam = np.sum(sample_data)/n
	print('lambda: {}'.format(lam))

	# initial guess for \lambda
	x0 = 1
	nll = poisson(x0, sample_data)
	print('nLL(data|lambda=1): {}'.format(nll))

	# minimize the negative log-likelihood
	result = optimize.minimize(fun = poisson, 
		x0 = x0, 
		args = sample_data, 
		method = 'L-BFGS-B',
		#bounds = ((0, 10),),
		callback = callback,
		options = {'maxiter': 10000, 'disp': True})

	print('Solution: x = {}'.format(result.x))

	# plot the nLL
	x = np.arange(1, 6, 0.1)
	y = []
	for x0 in x:
		val = poisson(x0, sample_data)
		y.append(val)

	import matplotlib.pyplot as plt
	plt.plot(x, y, 'r-', label='nLL(x)')
	plt.vlines(result.x[0], -500, 1000, linestyle='--', label='x0')
	plt.ylim([-500, 1000])
	plt.title('Negative Log-Likelihood')
	plt.xlabel('x')
	plt.ylabel('nLL(x)')
	plt.legend(loc='best')
	#plt.savefig('curve.pdf')
	plt.show()


