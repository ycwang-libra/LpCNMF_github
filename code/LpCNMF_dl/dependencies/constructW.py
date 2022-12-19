from dependencies.EuDist2 import EuDist2
import random
import math
import numpy as np
from scipy import sparse
from dependencies.utils import MATLAB_Mat, ReplaceDiag

def constructW(fea,options):
# %	Usage:
# %	W = constructW(fea,options)
# %
# %	fea: Rows of vectors of data points. Each row is x_i
# %   options: Struct value in Matlab. The fields in options that can be set:
# %                  
# %           NeighborMode -  Indicates how to construct the graph. Choices
# %                           are: [Default 'KNN']
# %                'KNN'            -  k = 0
# %                                       Complete graph
# %                                    k > 0
# %                                      Put an edge between two nodes if and
# %                                      only if they are among the k nearst
# %                                      neighbors of each other. You are
# %                                      required to provide the parameter k in
# %                                      the options. Default k=5.
# %               'Supervised'      -  k = 0
# %                                       Put an edge between two nodes if and
# %                                       only if they belong to same class. 
# %                                    k > 0
# %                                       Put an edge between two nodes if
# %                                       they belong to same class and they
# %                                       are among the k nearst neighbors of
# %                                       each other. 
# %                                    Default: k=0
# %                                   You are required to provide the label
# %                                   information gnd in the options.
# %                                              
# %           WeightMode   -  Indicates how to assign weights for each edge
# %                           in the graph. Choices are:
# %               'Binary'       - 0-1 weighting. Every edge receiveds weight
# %                                of 1. 
# %               'HeatKernel'   - If nodes i and j are connected, put weight
# %                                W_ij = exp(-norm(x_i - x_j)/2t^2). You are 
# %                                required to provide the parameter t. [Default One]
# %               'Cosine'       - If nodes i and j are connected, put weight
# %                                cosine(x_i,x_j). 
# %               
# %            k         -   The parameter needed under 'KNN' NeighborMode.
# %                          Default will be 5.
# %            gnd       -   The parameter needed under 'Supervised'
# %                          NeighborMode.  Colunm vector of the label
# %                          information for each data point.
# %            bLDA      -   0 or 1. Only effective under 'Supervised'
# %                          NeighborMode. If 1, the graph will be constructed
# %                          to make LPP exactly same as LDA. Default will be
# %                          0. 
# %            t         -   The parameter needed under 'HeatKernel'
# %                          WeightMode. Default will be 1
# %         bNormalized  -   0 or 1. Only effective under 'Cosine' WeightMode.
# %                          Indicates whether the fea are already be
# %                          normalized to 1. Default will be 0
# %      bSelfConnected  -   0 or 1. Indicates whether W(i,i) == 1. Default 0
# %                          if 'Supervised' NeighborMode & bLDA == 1,
# %                          bSelfConnected will always be 1. Default 0.
# %            bTrueKNN  -   0 or 1. If 1, will construct a truly kNN graph
# %                          (Not symmetric!). Default will be 0. Only valid
# %                          for 'KNN' NeighborMode
# %    Examples:
# %
# %       fea = rand(50,15);
# %       options = [];
# %       options.NeighborMode = 'KNN';
# %       options.k = 5;
# %       options.WeightMode = 'HeatKernel';
# %       options.t = 1;
# %       W = constructW(fea,options);
# %       
# %       
# %       fea = rand(50,15);
# %       gnd = [ones(10,1);ones(15,1)*2;ones(10,1)*3;ones(15,1)*4];
# %       options = [];
# %       options.NeighborMode = 'Supervised';
# %       options.gnd = gnd;
# %       options.WeightMode = 'HeatKernel';
# %       options.t = 1;
# %       W = constructW(fea,options);
# %       
# %       
# %       fea = rand(50,15);
# %       gnd = [ones(10,1);ones(15,1)*2;ones(10,1)*3;ones(15,1)*4];
# %       options = [];
# %       options.NeighborMode = 'Supervised';
# %       options.gnd = gnd;
# %       options.bLDA = 1;
# %       W = constructW(fea,options);      
# %       
# %
# %    For more details about the different ways to construct the W, please
# %    refer:
# %       Deng Cai, Xiaofei He and Jiawei Han, "Document Clustering Using
# %       Locality Preserving Indexing" IEEE TKDE, Dec. 2005.
# %    
# %
# %    Written by Deng Cai (dengcai2 AT cs.uiuc.edu), April/2004, Feb/2006,
# %                                             May/2007
# %    Translated by Yicheng Wang  5th/Oct/2021

	bBinary = 0
	nSmp = fea.shape[0]
	maxM = 62500000 # 500M
	BlockSize = math.floor(maxM / (nSmp * 3))

	G = np.zeros([nSmp*(options['k']+1),3])
	for i in range(math.ceil(nSmp/BlockSize)):
		if i == math.ceil(nSmp/BlockSize) - 1:
			smpIdx = [x + i*BlockSize for x in range(nSmp-i*BlockSize)]
			dist = EuDist2(fea[smpIdx,:],fea,0)

			nSmpNow = len(smpIdx)
			dump = np.zeros([nSmpNow,options['k']+1])
			idx = np.zeros([nSmpNow,options['k']+1])
			for j in range(options['k']+1):
				dump[:,j] = np.min(dist,1)
				idx[:,j] = np.argmin(dist,1)
				temp = (idx[:,j])*nSmpNow + np.transpose(np.array([x for x in range(nSmpNow)]))
				temp = temp.astype(np.int64)
				dist = MATLAB_Mat(dist, temp, 1e100)

			dump = np.exp(-dump/(2*options['t']**2))

			G[i*BlockSize*(options['k']+1):nSmp*(options['k']+1)+1,0] = \
				np.squeeze(np.tile(np.expand_dims(np.array(smpIdx),1),[options['k']+1,1]),1)
			G[i*BlockSize*(options['k']+1):nSmp*(options['k']+1)+1,1] = (np.transpose(idx)).reshape(1,-1) # 拉直成1维
			if not bBinary:
				G[i*BlockSize*(options['k']+1):nSmp*(options['k']+1)+1,2] = (np.transpose(dump)).reshape(1,-1)
			else:
				G[i*BlockSize*(options['k']+1):nSmp*(options['k']+1)+1,2] = 1
		else:
			smpIdx = [i*BlockSize + x for x in range(BlockSize)]
			dist = EuDist2(fea[smpIdx,:],fea,0)

			nSmpNow = len(smpIdx)
			dump = np.zeros([nSmpNow,options['k']+1])
			idx = dump
			for j in range(options['k']+1):
				dump[:,j] = np.min(dist,1)
				idx[:,j] = np.argmin(dist,1)
				temp = np.transpose(idx[:,j]*nSmpNow+[x for x in range(nSmpNow)])
				temp = np.int64(temp)
				dist = MATLAB_Mat(dist, temp, 1e100)

			dump = np.exp(-dump/(2*options['t']**2))

			G[i*BlockSize*(options['k']+1):(i+1)*BlockSize*(options['k']+1),0] = \
				np.squeeze(np.tile(np.expand_dims(np.array(smpIdx),1),[options['k']+1,1]),1)
			G[i*BlockSize*(options['k']+1):(i+1)*BlockSize*(options['k']+1),1] = idx.reshape(1,-1)
			if not bBinary:
				G[i*BlockSize*(options['k']+1):(i+1)*BlockSize*(options['k']+1),2] = dump.reshape(1,-1)
			else:
				G[i*BlockSize*(options['k']+1):(i+1)*BlockSize*(options['k']+1),2] = 1
	# 410行 
	W = sparse.coo_matrix((G[:,2],(G[:,0],G[:,1])),shape = (nSmp,nSmp))
	W = ReplaceDiag(W.todense(), 0)
	W = np.maximum(W, np.transpose(W))

	return np.array(W)
