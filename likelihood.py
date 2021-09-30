import numpy as np
import pandas as pd
from scipy import optimize

np.random.seed(1234)
Nfeval = 1

def gaussian(x, args):
	mean = args[0]
	sd = args[1]
	# Calculate the negative log likelihood
	nll = -np.sum(stats.norm.logpdf(x, loc=mean, scale=sd))
	return nll

def callback(x):
	global Nfeval
	# x is the current parameter vector.
	print('{}	{}'.format(Nfeval, x))
	Nfeval += 1

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

	# generate poisson distributed data with \lambda = 3
	# we want to identify \lambda using numeric MLE
	n = 1000
	sample_data = np.random.poisson(lam=3, size=n)
	print(sample_data)

	# analytic form of \lambda from analytic MLE
	# cross check
	lam = np.sum(sample_data)/n
	print(lam)

	# initial guess for \lambda
	x0 = 1
	q = poisson(x0, sample_data)
	print(q)
	
	# minimize the nLL
	print('Iter	X1	X2	X3	f(X)')
	result = optimize.minimize(fun = poisson, 
		x0 = x0, 
		args = sample_data, 
		method = 'BFGS', # 'L-BFGS-B',
		#bounds = ((0, 10),),
		callback = callback,
		options = {'maxiter': 10000, 'disp': True})

	print("Solution: x=%f" % result.x)

