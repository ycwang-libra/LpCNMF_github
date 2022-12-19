# -*- coding: utf-8 -*-
"""
data generation and loader
"""
import numpy as np
from dependencies.utils import LogiMul1d, LogiReplace
from dependencies.constructW import constructW

class Load_Data():
	def __init__(self,root_fea_path, root_gnd_path):
		self.root_fea_path = root_fea_path
		self.root_gnd_path = root_gnd_path

	def output_data(self):
		fea = np.load(self.root_fea_path) # nSample, nFea
		gnd = np.load(self.root_gnd_path) # nSample
		return fea, gnd

def get_loader(config):
	dataset = config.dataset
	data_root_path = config.data_root_path
	root_fea_path = data_root_path+'Normed_'+str(dataset)+'_fea.npy'
	root_gnd_path = data_root_path+'Normed_'+str(dataset)+'_gnd.npy'
	dataset = Load_Data(root_fea_path, root_gnd_path) # fea, gnd
	return dataset.output_data()

def gen_shuffleClasses(gnd, config):
	nClass  = len(np.unique(gnd))
	if config.dataset == 'MNIST':
		shuffleClasses = np.random.permutation(nClass) # shuffle 0,1,2,...nclass-1
	else:
		shuffleClasses = np.random.permutation(nClass) + 1 # shuffle 1,2,...nclass
	return shuffleClasses

def gen_shuffleIndexes(k, config):
	nsample = k * config.nEach
	shuffleIndexes = np.random.permutation(nsample) # 1 * 500k  1 * 1000 shuffle0～999
	return shuffleIndexes

def prepare_data_label(fea, gnd, config, k, shuffleClasses):
	# --- prepare the data and label--------------- random shuffleClasses---
	# Input: fea, gnd: data and label
	# Output: orgfeaSet: dataset only has k classes, each class has nEach sample
	# 		  orggndSet: continue label 1,1,1,2,2,2,3,3,3,...k
	#         shuffleClasses: shuffle 1,2,...nclass for reconstruction
	# -------------------------------------------------------------------------
	nEach = config.nEach
	nSample  = nEach * k
	nDim    = fea.shape[1]
	index  = 0
	Samples = np.zeros([nSample,nDim])
	Labels = np.zeros([nSample], dtype = int)
	for cls in range(k): 
		idx = np.where(gnd == shuffleClasses[cls])[0] 
		sampleEach = fea[idx[0:nEach],:]
		Samples[index:index+nEach,:] = sampleEach 
		Labels[index:index+nEach]  = cls + 1
		index = index + nEach

	orgfeaSet = Samples.copy()
	orggndSet = Labels.copy()
	return orgfeaSet, orggndSet

def sample_label(orggndSet, percent):
	# --- sample to label ------------random 2 shufflelabel only influent the labeled flag---------
	# Input: orggndSet: dataset only has k classes
	#        percent: percent to labeled 
	# Output: orgsemiSplit: labeled flag, flag = true means labeled, vice versa
	# --------------------------------------------------------------------------------------
	orgsemiSplit = np.zeros(orggndSet.shape[0], dtype = bool) # nsample
	k = len(np.unique(orggndSet))
	l = 0 # the first l data points with label information
	for cls in range(k):
		idx = np.where(orggndSet == (cls + 1))[0] # nEach * 1 location of this class(continue)
		shufflelabel = np.random.permutation(len(idx)) # 1 * nEach shuffle 0～nEach-1
		nSmpperlabel = np.uint8(np.floor(percent*len(idx)))  # percent * nEach each class sample for labeled
		l = l + nSmpperlabel
		orgsemiSplit[idx[shufflelabel[0:nSmpperlabel]]] = True
	return orgsemiSplit

def shuffle_datalabel(orgfeaSet, orggndSet, orgsemiSplit, shuffleIndexes):
	# --- shuffle the data and label ------random 3 shuffleIndexes---------------
	# Input: dataset only has k classes
	#       shuffleIndexes # 1 * 500k shuffle 0～500k-1
	# Output: feaSet, gndSet, semiSplit data, label and label flag 
	# --------------------------------------------------------------------------------------
	# shuffle
	feaSet = orgfeaSet[shuffleIndexes,:].copy()
	gndSet = orggndSet[shuffleIndexes].copy()
	semiSplit = orgsemiSplit[shuffleIndexes].copy()
	return feaSet, gndSet, semiSplit

def Necessary_Matrix(feaSet, gndSet, semiSplit, options, k):
	# ----------- constructing the necessary matrix for LpCNMF: A_lpcnmf, W, S -------------
	#  constructing matrix A_mid S
	nsample = feaSet.shape[0]
	S = np.diag(semiSplit) # bool semiSplit(500k * 1 logical) diagonal metirx 500k * 500k
	E = np.eye(k,dtype = int) # k * k diagonal metirx
	logimul = LogiMul1d(gndSet,semiSplit)
	A_mid = E[:,logimul-1] # k * (500k*percent) each sample for one column 10 for 1st class 01 for 2nd class 

	# constructing the label constraint matrix A_lpcnmf for lpcnmf
	A_lpcnmf = np.zeros([k,nsample])
	A_lpcnmf = LogiReplace(A_lpcnmf,semiSplit,A_mid,'row')

	# construction weight matrix
	W = constructW(feaSet,options)
	return A_lpcnmf, W, S

def reconstruct_data_label(fea, gnd, config, k):
	# 1. load the saved random vector
	rand_path = config.save_root_path + '/pretrain/UV_rand/'+ config.dataset  + '/'
	shuffleClasses_path = rand_path + 'shuffleClasses_{}.npy'.format(k)
	shuffleIndexes_path = rand_path + 'shuffleIndexes_{}.npy'.format(k)
	orgsemiSplit_path = rand_path + 'orgsemiSplit_{}.npy'.format(k)
	shuffleClasses = np.load(shuffleClasses_path)
	shuffleIndexes = np.load(shuffleIndexes_path)
	orgsemiSplit = np.load(orgsemiSplit_path)

	# 2. prepare_data_label
	orgfeaSet, orggndSet = prepare_data_label(fea, gnd, config, k, shuffleClasses)
	feaSet, gndSet, semiSplit = shuffle_datalabel(orgfeaSet, orggndSet, orgsemiSplit, shuffleIndexes)

	return feaSet, gndSet, semiSplit