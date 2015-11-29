#!/usr/bin/python

# original code is here:
#https://github.com/stephencwelch/Neural-Networks-Demystified/blob/master/partFour.py
# thanks to Stephen Welch @stephencwelch

import numpy as np
from scipy import optimize

# X = (hours sleeping, hours studying), y = Score on test
X = np.array(([3,5], [5,1], [10,2]), dtype=float)
y = np.array(([75], [82], [93]), dtype=float)

X = X/np.amax(X, axis=0)
y = y/100 #Max test score is 100


class NeuralNetwork(object):
	def __init__(self, Lambda=0):
		self.inputLayerSize = 2
		self.hiddenLayerSize=3
		self.outputLayerSize=1
		self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
		self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)
		self.Lambda = Lambda

	def sigmoid(self,z):
		return  1.0 / (1.0 + np.exp(-z));
	
	def sigmoidPrime(self,z):
		return np.exp(-z)/((1+np.exp(-z))**2)

	def costFunction(self,X,y):
		self.yhat = self.forward(X)
		return 0.5 * np.sum( (y-self.yhat)**2 ) + (self.Lambda/2)*(sum(self.W1**2)+sum(self.W2**2))
	 	
	def forward(self,X):
		# X -> 3x2, W1 -> 2x3
		self.z2 = np.dot(X,self.W1)
		self.a2 = self.sigmoid(self.z2)
		self.z3 = np.dot(self.a2,self.W2)
		self.yhat =  self.sigmoid(self.z2)
		return self.yhat
	
	def costFunctionPrime(self, X, y):
		self.yhat = self.forward(X)
		delta3 = np.multiply(-(y-self.yhat), self.sigmoidPrime(self.z3))
		dJdW2 = np.dot(self.a2.T, delta3) / X.shape[0] + self.Lambda*self.W2
		
		delta2 = np.dot(delta3, self.W2)*self.sigmoidPrime(self.z2)
		dJdW1 = np.dot(X.T, delta2) / X.shape[0] + self.Lambda*self.W1
		return dJdW1, dJdW2
	
	def gradients(self, X,y):
		djdw1,djdw2  = self.costFunctionPrime(X,y)
		return np.concatenate( (djdw1.ravel(),djdw2.ravel()) )

	def rollWeights(self):
		print self.W1.shape, self.W2.shape
		return np.concatenate( (self.W1.ravel(),self.W2.ravel()) )

	def unrollWeights(self, weights):
		w1start = 0
		w1end = self.inputLayerSize * self.hiddenLayerSize
		self.W1 = np.reshape(weights[w1start:w1end], (self.inputLayerSize, self.hiddenLayerSize))
		w2end = weights.shape[0]
		self.W2 = np.reshape(weights[w1end:w2end], (self.hiddenLayerSize, self.outputLayerSize))


	def numericalGradient(N, X,y):
		weights = N.rollWeights()
		EPSILONMATRIX = np.zeros(weights.shape[0])
		numgrad = np.zeros(weights.shape[0])
		EPSILON = 1e-4 
		
		for p in range(weights.shape[0]):
			EPSILONMATRIX[p] = EPSILON

			N.unrollWeights (weights + EPSILONMATRIX) 
			loss2 = N.costFunction(X,y)
			
			N.unrollWeights (weights - EPSILONMATRIX) 
			loss2 = N.costFunction(X,y)

			numgrad[p] = (loss2 - loss1) / (2.0*EPSILON)
			EPSILONMATRIX[p] = 0
		N.unrollWeights(weights)
	

class NNTrainer(object):
	def __init__(self, N):
		self.N = N

	def decoratedCostFunction(self, weights, X, y):
		self.N.unrollWeights(weights)
		cost = self.N.costFunction(X,y)
		grad = self.N.gradients(X,y)
		return cost, grad
	def callbackF(self, weights):
		self.N.unrollWeights(weights)
		self.J.append(self.N.costFunction(self.X,self.y))
	
	def train(self, trainX,trainY, testX, testY):
		#Make an internal variable for the callback function:
		self.X = trainX
		self.y = trainY

		self.testX = testX
		self.testY = testY

		#Make empty list to store training costs:
		self.J = []
		self.testJ = []

		weights = self.N.rollWeights()
		options = {'maxiter': 200, 'disp' : True}	
		res = optimize.minimize(self.decoratedCostFunction, weights, jac=True, method='BFGS',  args=(trainX, trainY), options=options, callback=self.callbackF)
		
		self.N.unrollWeights(res.x)
		self.opitimizationResults = res


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

	NN = NeuralNetwork()
	T = NNTrainer(NN)
	T.train(trainX, trainY, testX,testY)
		
	




run()

#nn = NeuralNetwork()
#djdw1, djdw2 = nn.costFunctionPrime(X,y)

#print djdw1
#print "\n\n"
#print djdw2


		
		

