#!/usr/bin/python

from math import *
import os.path
import sys, traceback
import re
import json
import copy 
class Utility:

	@staticmethod
	def log2(x):
		if x==0:
			return 0
		return log(x,2)

	""" uncertainity, information loss """
	@staticmethod
	def entropy(x,y):
		px = 1.0*x/(x+y)
		py = 1-px;
		result = - px * Utility.log2(px) - py * Utility.log2(py)
		return result

	#takes array and calculate entropy
	@staticmethod
	def entropyy(arr):
		s=0
		for x in arr:
			s+=x
		for i in xrange(0,len(arr)):
			arr[i] = 1.0 * arr[i]/s
		result = 0
		for x in arr:
			result-=x * Utility.log2(x)
		return result
		

class	Index:
	Outlook=0
	Temperature=1
	Humidity=2
	Wind=3
	PlayTennis=4


class NodeInfo:
	nodeValues=[]
	nodeEntropy=0
	children=[]
	frequencyCount=0
	def __init__(self,nodeValues):
		self.nodeValues=nodeValues
		self.calculateNodeEntropies()


	def calculateNodeEntropies(self):
		pass
		arr=[]
		for j in xrange(0, len(self.nodeValues)):
			if len(self.nodeValues[j])==0:
				break
			arr.append(self.nodeValues[j][1])
		self.nodeEntropy =  Utility.entropyy(arr)

	def findSameNodeNominal(self,nominal, arr):
		for i in xrange(0, len(arr)):
			if nominal== arr[0]:
				return i
		return -1

	def sumChildrenNodeValues(self):
		pass
		
		root = copy.deepcopy(self.children[0])
		for nodeValue in root.nodeValues:
			for i in xrange(1,len(self.children)):
				for j in xrange(0,len(self.children[i].nodeValues)):
					if nodeValue[0]==self.children[i].nodeValues[j][0]:
						nodeValue[1]+=self.children[i].nodeValues[j][1];
		
		self.frequencyCount=self.sumFreq(root.nodeValues)
		return root.nodeValues

	def sumFreq(self,arr):
		s=0
		for x in arr:
			s+=x[1]
		return s

	def gain(self):
		pass
		self.nodeValues = self.sumChildrenNodeValues()
		self.calculateNodeEntropies()
		gain=self.nodeEntropy
		print "root node entropy:"+ str(gain)
		for child in self.children:
			
			freq=self.sumFreq(child.nodeValues) 
			child_gain = 1.0*freq/self.frequencyCount * child.nodeEntropy 
			print "child.gain :" + str(freq) + "/" + str(self.frequencyCount) + " * "+  str(child.nodeEntropy) +" = "+ str(child_gain )
			gain -= child_gain
		return gain	

	def __str__(self):
	     return "{\nnodeValues:"+ json.dumps(self.nodeValues)+"\n" + "entropy: " + str(self.nodeEntropy) +"\n}\n"

class NodeInformationGain:
	nodeArray=[]
	resultArray=[]
	#nominalEntropies=[] # array of array. arr[0]=[nominal(x), prob_of_x, 1-prob_of_x]
	def __init__(self,  nodeArray, resultArray):
		pass
		self.nodeArray = nodeArray
		self.resultArray = resultArray

	def findNominalCount(self):
		myset = set(self.nodeArray)
		return len(myset)
	
	def nominalIndexIn(self,nominal, nominalEntropies):
		for i in xrange(0,len (nominalEntropies)):
			x = nominalEntropies[i]
			if nominal==x[0]:
				return i
		return -1

	def calculateNominalEntropies(self, nominal,nodeArray, resultArray):
		nominalEntropies=[]
		for i in xrange(0, len(nodeArray)):
			x = nodeArray[i]
			if x!=nominal:
				continue
			idx = self.nominalIndexIn(resultArray[i], nominalEntropies)
			if idx==-1:
				nominalEntropies.append([resultArray[i],1])
			else:
				count = nominalEntropies[idx][1] + 1
				nominalEntropies[idx][1]=count	

		# nomnalEntropies will have below format
		# nomianlEntropies[0][2]
		# nomianlEntropies[1][3]
		# accordingly we have 2 nominal values:0,1 and probabilities are 2/5 and 3/5

		return nominalEntropies
		#arr=[]
		#for j in xrange(0, len(nominalEntropies)):
		#	arr.append(nominalEntropies[j][1])
		#return  Utility.entropyy(arr)
			


	def gain(self):
		myset = set(self.nodeArray)
		nomCount = len (myset) # count of different nominals
		nodeValues=[]
		children=[]
		node=NodeInfo([[],[]])
		for nominal in myset: 
			nodeValue = self.calculateNominalEntropies(nominal,self.nodeArray, self.resultArray)
			ni  = NodeInfo(nodeValue)
			children.append(ni)
		node.children=children;
		g = node.gain();
		return g

class DecisionTree:
	data=[]
	filePath=""
	def readData(self):
		pass
		if os.path.exists(self.filePath)==False:
			print "file does not exist:" + self.filePath +"\n"
			sys.exit(-1)
		file = open(self.filePath, 'r')
		file.readline()
		for line in file:
			lineTokens = re.split(r'\s+',line)
			del lineTokens[0]
			del lineTokens[5]
			self.data.append(lineTokens)
		file.close()
	def gain(self,nodeIndex,resultIndex):
		nodeArray=[]
		resultArray=[]

		for i in xrange(0, len(self.data)):
			nodeArray.append( self.data[i][nodeIndex])

		for i in xrange(0, len(self.data)):
			resultArray.append( self.data[i][resultIndex])
		nig = NodeInformationGain(nodeArray,resultArray).gain()
		return nig

	def __init__(self,filePath):
		pass
		self.filePath = filePath
		self.readData()




dt = DecisionTree("data.txt")
gain = dt.gain(Index.Wind,Index.PlayTennis)
print "root node gain:" + str(gain)
