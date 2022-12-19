# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 21:44:30 2021

@author: yc_wang
"""
from scipy.io import loadmat
import numpy as np
import random
from scipy import sparse
from dependencies.utils import MATLAB233, LogiMul1d

def litekmeans(X, k, var, reps):
# %LITEKMEANS K-means clustering, accelerated by matlab matrix operations.
# %
# %   label = LITEKMEANS(X, K) partitions the points in the N-by-P data matrix
# %   X into K clusters.  This partition minimizes the sum, over all
# %   clusters, of the within-cluster sums of point-to-cluster-centroid
# %   distances.  Rows of X correspond to points, columns correspond to
# %   variables.  KMEANS returns an N-by-1 vector label containing the
# %   cluster indices of each point.
# %
# %   [label, center] = LITEKMEANS(X, K) returns the K cluster centroid
# %   locations in the K-by-P matrix center.
# %
# %   [label, center, bCon] = LITEKMEANS(X, K) returns the bool value bCon to
# %   indicate whether the iteration is converged.  
# %
# %   [label, center, bCon, SUMD] = LITEKMEANS(X, K) returns the
# %   within-cluster sums of point-to-centroid distances in the 1-by-K vector
# %   sumD.    
# %
# %   [label, center, bCon, SUMD, D] = LITEKMEANS(X, K) returns
# %   distances from each point to every centroid in the N-by-K matrix D. 
# %
# %   [ ... ] = LITEKMEANS(..., 'PARAM1',val1, 'PARAM2',val2, ...) specifies
# %   optional parameter name/value pairs to control the iterative algorithm
# %   used by KMEANS.  Parameters are:
# %
# %   'Distance' - Distance measure, in P-dimensional space, that KMEANS
# %      should minimize with respect to.  Choices are:
# %            {'sqEuclidean'} - Squared Euclidean distance (the default)
# %             'cosine'       - One minus the cosine of the included angle
# %                              between points (treated as vectors). Each
# %                              row of X SHOULD be normalized to unit. If
# %                              the intial center matrix is provided, it
# %                              SHOULD also be normalized.
# %
# %   'Start' - Method used to choose initial cluster centroid positions,
# %      sometimes known as "seeds".  Choices are:
# %         {'sample'}  - Select K observations from X at random (the default)
# %          'cluster' - Perform preliminary clustering phase on random 10%
# %                      subsample of X.  This preliminary phase is itself
# %                      initialized using 'sample'. An additional parameter
# %                      clusterMaxIter can be used to control the maximum
# %                      number of iterations in each preliminary clustering
# %                      problem.
# %           matrix   - A K-by-P matrix of starting locations; or a K-by-1
# %                      indicate vector indicating which K points in X
# %                      should be used as the initial center.  In this case,
# %                      you can pass in [] for K, and KMEANS infers K from
# %                      the first dimension of the matrix.
# %
# %   'MaxIter'    - Maximum number of iterations allowed.  Default is 100.
# %
# %   'Replicates' - Number of times to repeat the clustering, each with a
# %                  new set of initial centroids. Default is 1. If the
# %                  initial centroids are provided, the replicate will be
# %                  automatically set to be 1.
# %
# % 'clusterMaxIter' - Only useful when 'Start' is 'cluster'. Maximum number
# %                    of iterations of the preliminary clustering phase.
# %                    Default is 10.  
# %
# %
# %    Examples:
# %
# %       fea = rand(500,10);
# %       [label, center] = litekmeans(fea, 5, 'MaxIter', 50);
# %
# %       fea = rand(500,10);
# %       [label, center] = litekmeans(fea, 5, 'MaxIter', 50, 'Replicates', 10);
# %
# %       fea = rand(500,10);
# %       [label, center, bCon, sumD, D] = litekmeans(fea, 5, 'MaxIter', 50);
# %       TSD = sum(sumD);
# %
# %       fea = rand(500,10);
# %       initcenter = rand(5,10);
# %       [label, center] = litekmeans(fea, 5, 'MaxIter', 50, 'Start', initcenter);
# %
# %       fea = rand(500,10);
# %       idx=randperm(500);
# %       [label, center] = litekmeans(fea, 5, 'MaxIter', 50, 'Start', idx(1:5));
# %
# %
# %   See also KMEANS
# %
# %    [Cite] Deng Cai, "Litekmeans: the fastest matlab implementation of
# %           kmeans," Available at:
# %           http://www.zjucadcg.cn/dengcai/Data/Clustering.html, 2011. 
# %
# %   version 2.0 --December/2011
# %   version 1.0 --November/2011
# %
# %   Written by Deng Cai (dengcai AT gmail.com)
# %   Translated by Yicheng Wang on November/2021

	n, p = X.shape

	distance = 'sqeuclidean'
	start = 'sample'
	maxit = 100
	clustermaxit = 10
	bestlabel = np.matrix(0)
	sumD = np.zeros([1,k])
	bCon = False
	
	for t in range(reps):
		center = X[np.random.choice(range(n),k,replace = False),:] # initial center

		last = np.array([[0]])
		label = np.array([[1]])
		it = 0

		while any(label != last) and it < maxit:
			last = label.copy()
			bb = (np.sum(center*center, axis = 1)).reshape(1,-1)
			ab = np.dot(X, np.transpose(center))  # 1000 * 2
			D = bb[np.zeros([n],dtype = 'int32'),:] - 2 * ab
			val = np.min(D,1)  # 1 * n
			label = (np.argmin(D,1)).reshape(-1,1)
			ll = np.unique(label)
			if len(ll) < k:
				missCluster = np.array(range(k))
				missCluster = np.delete(missCluster,ll)
				missNum = len(missCluster)

				aa = np.sum(X * X, 1) # n * p --> 1 * n
				val = aa + val
				idx = np.argsort(-val)
				label[idx[:missNum]] = missCluster.reshape(-1,1)
			
			E = MATLAB233(label,n,k,1) # transform label into indicator matrix
		
			center = np.array(np.dot(np.transpose(np.dot(E,(sparse.spdiags(1/np.sum(E,0),0,k,k)).todense())),X))
			it = it + 1

		if it < maxit:
			bCon = True

		if not bestlabel.any():
			bestlabel = label.copy()
			bestcenter = center.copy()
			if reps > 1:
				if it >= maxit:
					aa = (np.sum(X * X, 1)).reshape(-1,1) # n x k --> n x 1
					bb = (np.sum(center * center,1)).reshape(-1,1) # k x k --> k x 1
					ab = np.dot(X, np.transpose(center)) # n x k
					D = aa + np.transpose(bb) - 2 * ab
					D[D< 0] = 0
				else:
					aa = (np.sum(X * X, 1)).reshape(-1,1)   # 1000 x 1
					D = aa[:,np.zeros([k],dtype = 'int32')] + D
					D[D< 0] = 0

				D = np.sqrt(D)
				for j in range(k):
					sumD[0,j] = np.sum(LogiMul1d(D,label == j)[:,j])
				bestsumD = sumD.copy()
				bestD = D.copy()
		else:
			if it >= maxit:
				aa = (np.sum(X * X, 1)).reshape(-1,1)
				bb = (np.sum(center * center,1)).reshape(-1,1)
				ab = np.dot(X, np.transpose(center))
				D = aa + np.transpose(bb) - 2 * ab
				D[D< 0] = 0
			else:
				aa = (np.sum(X * X, 1)).reshape(-1,1)
				D = aa[:,np.zeros([k],dtype = 'int32')] + D
				D[D< 0] = 0

			D = np.sqrt(D)
			for j in range(k):
				sumD[0,j] = np.sum(LogiMul1d(D,label == j)[:,j])
			if np.sum(sumD) < np.sum(bestsumD):
				bestlabel = label.copy()
				bestcenter = center.copy()
				bestsumD = sumD.copy()
				bestD = D.copy()
	
	label = bestlabel.copy()
	center = bestcenter.copy()
	if reps > 1:
		sumD = bestsumD.copy()
		D = bestD.copy()

	label = np.array(label)
	return label, center, bCon, sumD, D