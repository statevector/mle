import numpy as np
import pandas as pd
from scipy import optimize

np.random.seed(1234)

def gaussian(x, args):
	mean = args[0]
	sd = args[1]
	# Calculate the negative log likelihood
	nll = -np.sum(stats.norm.logpdf(x, loc=mean, scale=sd))
	return nll

def callback(x):
	'''
    Callback called after each iteration.
    x is the current estimated location of the objective minimum.
	'''
	print(x)

# x-hat = k/n
def binomial(x, args):
    # Calculate the negative log likelihood
	k, n = args[0], args[1]
	#print('k: {}, n: {}'.format(k, n))
	nLL = -(k*np.log(x) + (n-k)*np.log(1-x))
	#print('nLL: {}'.format(nLL))
	return nLL

# x-hat = sum(k)/n
def poisson(x, args):
	# Calculate the negative log likelihood
	#k = args[0]
	#n = len(args)
	k = args
	# print(k)
	# print(np.sum(k))
	# print(np.log(x))
	# print(len(k))
	# print(x)
	nLL = -(np.sum(k)*np.log(x) - len(k)*x)
	# print(nLL)
	return nLL

if __name__ == '__main__':

	# generate poisson distributed data for \lambda = 3
	# we want to identify \lambda using MLE
	n = 1000
	sample_data = np.random.poisson(lam=3, size=n)
	print(sample_data)
	print(np.sum(sample_data)/n)

	# initial guess for \lambda
	x0 = 1
	poisson(x0, sample_data)

	# minimize the nLL
	result = optimize.minimize(fun = poisson, 
		x0 = x0, 
		args = sample_data, 
		method = 'L-BFGS-B',
		bounds = ((0, 10),),
		callback = callback,
		options = {'maxiter': 10000, 'disp':True})

	print("Solution: x=%f" % result.x)

