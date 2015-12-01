#!/usr/bin/python



import numpy as np
import matplotlib.pyplot as plt
import pylab
import random
from numpy import linalg as LA

#idxs = [ x for x in range(0,X.shape[0])]
#random.shuffle(idxs)


class Cluster(object):

	def __init__(self,points=[]):
		self.points = points
		self.count = len(self.points)
		if len(points)!=0:
			self.calcCentroid()

	def setCentroid(self,point):
		self.centroid = point

	def getCentroid(self):
		return self.centroid
	
	def calcCentroid(self):
		x,y = np.mean(self.points, axis=0) 
		self.centroid = [x,y]
		return np.array(self.centroid)

	def update(self, cluster):
		distanceToOldCluster = self.distanceTo(cluster)
		self.points = cluster.points
		self.count = len(self.points)
		return distanceToOldCluster, self.calcCentroid()

	def distanceTo(self, cluster):
		return LA.norm(self.centroid -  cluster.centroid)

	def append(self, p):
		self.points.append(p)

class KMeans(object):
	
	def __init__(self):
		self.clusterSize = 100
		self.pt1 = np.random.normal(1, 0.3, (self.clusterSize,2))
		self.pt2 = np.random.normal(2, 0.3, (self.clusterSize,2))
		self.pt3 = np.random.normal(3, 0.5, (self.clusterSize,2))
		self.X = np.array(np.concatenate((self.pt1, self.pt2, self.pt3)))
		self.m,self.n = self.X.shape
		self.optimizationCutoff = 0.5
		self.clusters = []


	def plot(self):
		colors = ['yo','go','bo']
		for s in range(len(colors)):
			start = s*100
			end = start+100
			plt.plot(self.X[start:end,0],self.X[start:end,1], colors[s])

		for i,c in enumerate(self.clusters):
			print c.getCentroid()
			plt.plot([c.getCentroid()[0]], [c.getCentroid()[1]], 'rx', mew=10, ms=20)
		plt.show()


	def createInitialClustersWithRandomCentroids(self,k):
		randIdxs = np.random.randint(300,size=k)
		for i in range(len(randIdxs)):
			point = self.X[i]
			cluster = Cluster()
			cluster.setCentroid(point)
			self.clusters.append(cluster) 
		

	def assignPointsToNearestClusterAccordingToClusterCentroids(self):
		for x in self.X:
			min_dist=9999999
			min_idx=0
			#find which cluster is nearest to the point x
			for cidx,c in enumerate(self.clusters):
				dist = LA.norm( x - c.getCentroid() ) 
				if dist<min_dist:
					min_dist = dist
					min_idx=cidx
			#min_idx stores the nearest cluster's index
			#assign x to the nearest cluster
			self.clusters[min_idx].append(x)
		
	def recalculateCentroids(self):
		for i, c in enumerate(self.clusters):
			c.calcCentroid()

		
	def clearClusterPoints(self):
		for i,c in enumerate(self.clusters):
			c.points=[]

	#for debugging issues
	def printCentroids(self):
		for i,o in enumerate(self.clusters):
			print "centroid: " + str(o.getCentroid())
			

	def kmeans(self,k):
		
		self.clusterCount = k
		self.createInitialClustersWithRandomCentroids(self.clusterCount)
		print "cluster's initial centroids:"
		for i,o in enumerate(self.clusters):
			print (self.clusters[i].getCentroid())
		loopCounter=0
		for  i in range(15):
			loopCounter+=1
			print "loop:"+ str(loopCounter)
			self.clearClusterPoints()
			self.assignPointsToNearestClusterAccordingToClusterCentroids()	
			self.recalculateCentroids()
		
		
		

	def run(self,k):
		np.random.seed(99)
		clusters = self.kmeans(k)
		self.printCentroids()
		self.plot()



kmeans = KMeans()
kmeans.run(3)
