# Euclidean Clustering Module
# author Aaron Brown
# modified by W.Jin and S.Lim

import numpy as np
from sklearn.neighbors import KDTree

def clusterHelper(index, points, cluster, processed, tree, distanceTol):
	processed[index] = True
	cluster.append(index)
	nearest_index, nearest_distance = tree.query_radius([points[index]], r=distanceTol, return_distance=True)
	for i in range(len(nearest_index[0])):
		idx = nearest_index[0][i]
		if processed[idx] == False:
			clusterHelper(idx, points, cluster, processed, tree, distanceTol)

def euclideanCluster(points, tree, distanceTol):
	clusters = []
	processed = [False for i in range(len(points))]
	i = 0
	while i < len(points):
		if processed[i] == True:
			i += 1
			continue
		cluster = []
		clusterHelper(i, points, cluster, processed, tree, distanceTol)

		if(len(cluster) > 10):
			clusters.append(cluster)
			
		i += 1
	
	return clusters

if __name__ == "__main__":
    print("Error.. Why clusteringModule execute")