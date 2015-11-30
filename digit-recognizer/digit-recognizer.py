#!/usr/bin/python
# Neural Networks Demystified
# Part 6: Training
#
# Supporting code for short YouTube series on artificial neural networks.
#
# Stephen Welch
# @stephencwelch


import numpy as np
from scipy import optimize, io
from pylab import imshow
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import random




class Neural_Network(object):
	def __init__(self, Lambda=0):        
		#Define Hyperparameters
		self.inputLayerSize = 400
		self.outputLayerSize = 4
		self.hiddenLayerSize = 8

		#Weights (parameters)
		self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
		self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)
		self.Lambda=Lambda

	def forward(self, X):
		#Propogate inputs though network
		self.z2 = np.dot(X, self.W1)
		self.a2 = self.sigmoid(self.z2)
		self.z3 = np.dot(self.a2, self.W2)
		yHat = self.sigmoid(self.z3) 
		return yHat

	def sigmoid(self, z):
		#Apply sigmoid activation function to scalar, vector, or matrix
		return 1/(1+np.exp(-z))

	def sigmoidPrime(self,z):
		#Gradient of sigmoid
		return np.exp(-z)/((1+np.exp(-z))**2)

	def costFunction(self, X, y):
		#Compute cost for given X,y, use weights already stored in class.
		self.yHat = self.forward(X)
		J = 0.5*np.sum((y-self.yHat)**2)/X.shape[0] + (self.Lambda/2)*(np.sum(self.W1**2)+np.sum(self.W2**2))
		return J

	def costFunctionPrime(self, X, y):
		#Compute derivative with respect to W and W2 for a given X and y:
		self.yHat = self.forward(X)

		delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
		dJdW2 = np.dot(self.a2.T, delta3)/X.shape[0] + self.Lambda*self.W2


		delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
		dJdW1 = np.dot(X.T, delta2)/X.shape[0] + self.Lambda*self.W1


		return dJdW1, dJdW2

	#Helper Functions for interacting with other classes:
	def rollWeights(self):
		#Get W1 and W2 unrolled into vector:
		weights = np.concatenate((self.W1.ravel(), self.W2.ravel()))
		return weights

	def unrollWeights(self, weights):
		#Set W1 and W2 using single paramater vector.
		W1_start = 0
		W1_end = self.hiddenLayerSize * self.inputLayerSize
		self.W1 = np.reshape(weights[W1_start:W1_end], (self.inputLayerSize , self.hiddenLayerSize))
		W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
		self.W2 = np.reshape(weights[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))

	def computeGradients(self, X, y):
		dJdW1, dJdW2 = self.costFunctionPrime(X, y)
		return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))

	def computeNumericalGradient(N, X, y):
		weightsInitial = N.rollWeights()
		numgrad = np.zeros(weightsInitial.shape)
		perturb = np.zeros(weightsInitial.shape)
		e = 1e-4

		for p in range(len(weightsInitial)):
			#Set perturbation vector
			perturb[p] = e
			N.unrollWeights(weightsInitial + perturb)
			loss2 = N.costFunction(X, y)

			N.unrollWeights(weightsInitial - perturb)
			loss1 = N.costFunction(X, y)

			#Compute Numerical Gradient
			numgrad[p] = (loss2 - loss1) / (2*e)

			#Return the value we changed to zero:
			perturb[p] = 0

			#Return Params to original value:
			N.unrollWeights(weightsInitial)

		return numgrad 



class trainer(object):
	TrainCount=0
	def __init__(self, N):
		#Make Local reference to network:
		self.N = N

	def callbackF(self, weights):
		#self.N.unrollWeights(weights)
		#self.J.append(self.N.costFunction(self.X, self.y))   
		print self.TrainCount
		trainer.TrainCount +=1

	def costFunctionWrapper(self, weights, X, y):
		self.N.unrollWeights(weights)
		cost = self.N.costFunction(X, y)
		grad = self.N.computeGradients(X,y)
		return cost, grad


	def train(self, trainX, trainY, testX, testY):
		#Make an internal variable for the callback function:
		#self.X = trainX
		#self.y = trainY

		#self.testX = testX
		#self.testY = testY

		#Make empty list to store training costs:
		self.J = []
		self.testJ = []

		weights = self.N.rollWeights()

		options = {'maxiter': 20, 'disp' : True}
		_res = optimize.minimize(self.costFunctionWrapper, weights, jac=True, method='BFGS',  args=(trainX, trainY), options=options, callback=self.callbackF)

		self.N.unrollWeights(_res.x)
		self.optimizationResults = _res
		return self.N.W1,self.N.W2


def run():
	#Training Data:
	trainX = np.array(([3,5], [5,1], [10,2], [6,1.5]), dtype=float)
	trainY = np.array(([75], [82], [93], [70]), dtype=float)

	#Testing Data:
	testX = np.array(([4, 5.5], [4.5,1], [9,2.5], [6, 2]), dtype=float)
	testY = np.array(([70], [89], [85], [75]), dtype=float)

	#Normalize:
	trainX = trainX/np.amax(trainX, axis=0)
	trainY = trainY/100 #Max test score is 100

	#Normalize by max of training data:
	testX = testX/np.amax(trainX, axis=0)
	testY = testY/100 #Max test score is 100



	#Train network with new data:
	NN = Neural_Network(Lambda=0.01)
	T = trainer(NN)
	trainedWeights = T.train(trainX, trainY, testX, testY)
	#print trainedWeights

	NN.W1 = trainedWeights[0]
	NN.W2 = trainedWeights[1]
	yhat = NN.forward(np.array( ([6,8]), dtype=float   ))
	print yhat


class DigitRec(object):
	
	def __init__(self):
		self.test=1	
	

	def plotData(self, c):
		img = self.X[c,:]
		imgdata = img.reshape(20,20).T
		print self.y[c][0]
		plt.imshow(imgdata, interpolation='nearest', cmap = cm.Greys_r)
		plt.show()
	

	def loadData(self):
		print "loading the data"
		filePath = "/home/halil/root/machine-learning/andrew-ng-coursera/machine-learning-ex4/ex4/ex4data1.mat"
		data  = io.loadmat(filePath)
		self.X = data [ 'X' ]
		self.X = self.X[0:2000,]
		self.y = data [ 'y' ]
		self.y = self.y[0:2000]
		self.y[self.y==10]=0
		self.y = self.kbasedNumber(self.y)
		#numOfLabels = np.unique(self.y).size
		#print numOfLabels

	def kbasedNumber(self,y):
		m,n = y.shape
		classes = np.unique(y).size
		y_ = np.zeros((m,classes))

		for i in range(0, m):
			y_[i, y[i]] = 1
		return y_

	def run(self):
		self.loadData()
		idxs = [ x for x in range(0,self.X.shape[0])]
		random.seed(99)
		random.shuffle(idxs)
		cutoff = int(len(idxs) * 0.7)
		trainX = np.array([self.X[i] for i in idxs[0:cutoff]])
		trainY = np.array([self.y[i] for i in idxs[0:cutoff]])
		testX = np.array([self.X[i] for i in idxs[cutoff:]])
		testY = np.array([self.y[i] for i in idxs[cutoff:]])

		print trainX.shape
		print trainY.shape
		NN = Neural_Network(Lambda=0.01)
		T = trainer(NN)
		trainedWeights = T.train(trainX, trainY, testX, testY)

		NN.W1 = trainedWeights[0]
		NN.W2 = trainedWeights[1]
		yhat = NN.forward(np.array( (testX[0]), dtype=float   ))
		print yhat
		print "cutoff:" + str(idxs[cutoff])
		print testY[0]
		#print self.y[1002]
		#self.plotData(1102)
		#print self.kbasedNumber(self.y)[1502]


#run()
DigitRec().run()
