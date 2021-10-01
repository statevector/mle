import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import norm

np.random.seed(1234)
Nfeval = 1

def callback(x):
	global Nfeval
	# x is the current parameter vector
	print('{}	{}'.format(Nfeval, x))
	Nfeval += 1

def gaussian(x, args):
	k = args
	mu = x[0]
	sd = x[1]
	# Calculate the negative log likelihood
	#nLL = -np.sum(norm.logpdf(k, loc=mu, scale=sd))
	nLL = 1/2*len(k)*np.log(2*np.pi*sd**2) + 1/(2*sd**2)*np.sum((k-mu)**2)
	#print(nLL)
	return nLL

if __name__ == '__main__':

	# generate gaussian distributed data with \mu = 2, \sigma = 2
	# we want to identify \mu and \sigma using numeric MLE
	n = 100
	sample_data = np.random.normal(loc=2, scale=2, size=n)
	print('data: {}'.format(sample_data))

	# analytic form of \mu
	mu = np.sum(sample_data)/n
	print('mu: {}'.format(mu))

	# analytic form of \sigma^2
	sigma2 = np.sum((sample_data - mu)**2)/n
	print('sigma2: {}'.format(sigma2))

	# initial guess for p
	x0 = [0.5, 0.5]
	nll = gaussian(x0, sample_data)
	print(nll)

	# minimize the negative log-likelihood
	result = optimize.minimize(fun = gaussian, 
		x0 = x0, 
		args = sample_data, 
		method = 'L-BFGS-B',
		# bounds = ((0, 1),),
		callback = callback,
		options = {'maxiter': 10000, 'disp': True})

	print('Solution: x = {}'.format(result.x))
	
	# plot the nLL
	import matplotlib.pyplot as plt
	xlist = np.arange(1.0, 3, 0.01)
	ylist = np.arange(1.0, 3, 0.01)
	X, Y = np.meshgrid(xlist, ylist)
		
	Z = []
	for ix, x in enumerate(xlist):
		Z.append([])
		for y in ylist:
			nll = gaussian([x, y], sample_data)
			Z[ix].append(nll)
	Z = np.array(Z)

	fig, ax = plt.subplots(1,1)
	cp = ax.contour(xlist, ylist, Z, levels=100)
	plt.plot(result.x[0], result.x[1], marker='*', lw=0, color='red', label='min(nLL)')
	plt.plot(2, 2, marker='*', lw=0, color='black', label='truth')
	fig.colorbar(cp) # Add a colorbar to a plot
	ax.set_title('Negative Log-Likelihood')
	ax.set_xlabel('mu')
	ax.set_ylabel('sigma')
	ax.legend(loc='best')
	plt.show()

