#!/usr/bin/python


import csv
import numpy as np
from numpy import genfromtxt
from numpy import loadtxt, where
from pylab import scatter, show, legend, xlabel, ylabel
from scipy.optimize import fmin_bfgs


def map_feature(x1, x2):
	x1.shape = (x1.size, 1)
	x2.shape = (x2.size, 1)
	degree = 6
	out = np.ones(shape=(x1[:, 0].size, 1))

	m, n = out.shape

	for i in range(1, degree + 1):
		for j in range(i + 1):
			r = (x1 ** (i - j)) * (x2 ** j)
			out = np.append(out, r, axis=1)

	return out


def sigmoid(z):
	return 1 / (1 + np.exp(-1 * z))

def log(z):
	return np.log(z)

def plotData():
	data = genfromtxt('../ex2/ex2data1.txt', delimiter=',')
	X = data[:, 0:2]
	y = data[:, 2]
	pos = where(y==1)
	neg = where(y==0)
	scatter(X[pos, 0], X[pos, 1], marker='o', c='b')
	scatter(X[neg, 0], X[neg, 1], marker='x', c='r')
	show()

	

def computeCost(theta,X,y,lambda_):
	print theta.shape
	print X.shape
	m = np.shape(y)[0]
	h = sigmoid(X.dot(theta))
	thetaR = theta[1:, 0]

	J = (1.0/m) *  (-1 * y.T).dot(  log(h) ) - (1-y).T.dot( log(1 - h) )  + (lambda_/(2.0*m)) * thetaR.T.dot(thetaR)
	delta = h - y
	sumdelta = delta.T.dot(X[:,1])
	grad1 = (1.0/m) * sumdelta

	XR = X[:, 1:X.shape[1]]
	sumdelta = delta.T.dot(XR)
	grad = (1.0 / m) * (sumdelta + lambda_ * thetaR)
	#out = np.zeros(shape=(grad.shape[0], 2))
	out = np.zeros(shape=(grad.shape[0], grad.shape[1] + 1))

	out[:, 0] = grad1
	out[:, 1:] = grad
	return J.flatten(), out.T.flatten()



	

data = genfromtxt('../ex2/ex2data1.txt', delimiter=',')
X = data[:, 0:2]
m,n = X.shape#
#ones = np.ones(m)
#X = np.c_[ones,X]
y = data[:, 2]
#it = map_feature(X[:, 0], X[:, 1])
it = X
initial_theta = np.ones(shape=(it.shape[1], 1))
lambda_=1

#cost, grad = computeCost(it,y,initial_theta,  lambda_)
def decorated_cost(theta__,it_,y_,lambda__):
	theta_ = np.ones(shape=(it_.shape[1], 1))
	theta_[:,0] = theta__[:]
	return computeCost(theta_, it_, y_,  lambda__)[0]

def decorated_gradient(theta__, it_, y_, lambda__):
	theta_ = np.ones(shape=(it_.shape[1], 1))
	theta_[:,0] = theta__[:]
	return computeCost(theta_, it_, y_,  lambda__)[1]

#[J, grad] = computeCost(X,y,theta,lambda_)
#print grad

print fmin_bfgs(decorated_cost, initial_theta, maxiter=400, args=(it, y, 1),  fprime=decorated_gradient  )



#plotData()
#z = sigmoid(np.array([1,2]))
#print z

