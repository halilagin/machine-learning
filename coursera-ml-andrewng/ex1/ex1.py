#!/usr/bin/python


import csv
import numpy as np
from numpy import genfromtxt





def computeCost(X,y,theta):
	m = np.shape(y)[0]
	preds = np.dot(X,theta)
	#fark = np.c_[preds,y]
	#print fark
	sqrErr = np.power(np.subtract(preds,y),2)
	return (1.0/(2*m) ) *  np.sum(sqrErr)


def gradientDecent(X,y,theta,alpha,iterations):
	m = y.shape[0]
	J_history = np.zeros(iterations)
	#Xt = np.transpose(X)
	for i in range(0, iterations):
		h = np.dot(X , theta )
		theta_zero = theta[0] -  alpha * (1.0/m) * np.sum( np.subtract(h,y) )  
		theta_one = theta[1] - alpha * (1.0/m) * np.sum( np.subtract(h,y) * X[:,1]  )
		theta = [theta_zero, theta_one]
		J_history[i] = computeCost(X,y,theta )
	return [J_history,theta]
	

data = genfromtxt('../ex1/ex1data1.txt', delimiter=',')
theta = np.zeros(2)
X = data[:,0]
y = data[:,1]

m = data.shape[0]
n = data.shape[1]
ones = np.ones(m)
X = np.c_[ones,X]

ret = gradientDecent(X,y, theta, 0.001, 1500)
print ret[1] # new theta, ret[0] is j_history
