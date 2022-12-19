# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 19:08:58 2021

@author: yc_wang
"""
import numpy as np
from dependencies.utils import LogiMerge, Mat2Array

def MutualInfo(L1, L2):
# %   mutual information
# %
# %   version 2.0 --May/2007
# %   version 1.0 --November/2003
# %
# %   Written by Deng Cai (dengcai AT gmail.com)
# %   Translated by Yicheng Wang at 2021-11-03
	L1 = L1.reshape(-1,1)
	L2 = L2.reshape(-1,1)
	if L1.size != L2.size:
		print('size(L1) must == size(L2)')
	Label = (np.unique(L1)).reshape(-1,1)
	nClass = len(Label)
	Label2 = (np.unique(L2)).reshape(-1,1)
	nClass2 = len(Label2)

	if nClass2 < nClass:
		# Smooth
		L1 = np.concatenate((L1, Label),axis = 0)
		L2 = np.concatenate((L2, Label),axis = 0)
	elif nClass2 > nClass:
		# Smooth
		L1 = np.concatenate((L1, Label2),axis = 0)
		L2 = np.concatenate((L2, Label2),axis = 0)

	G = np.zeros([nClass,nClass])
	for i in range(nClass):
		for j in range(nClass):
			gg = sum(LogiMerge(L1==Label[i],L2==Label[j],'and'))
			G[i,j] = gg if gg!=0 else 1e-10   # debugged Label2 --> Label
	
	sumG = sum(sum(G)) if sum(sum(G))!=0 else 1e-10

	P1 = (np.sum(G,1)).reshape(-1,1)
	P1 = P1 / sumG
	P2 = np.sum(G,0).reshape(1,-1)
	P2 = P2 / sumG
	if sum(P1==0) > 0 and sum(P2==0) > 0:
		print('Smooth fail!')
	else:
		H1 = sum(sum(-P1 * np.log2(P1)))
		H2 = sum(sum(-P2 * np.log2(P2)))
		P12 = G / sumG
		PPP = P12 / np.tile(P2,(nClass,1)) / np.tile(P1,(1,nClass))
		PPP[abs(PPP)<1e-12] = 1
		MI = sum(Mat2Array(P12, ID = 0) * np.log2(Mat2Array(PPP, ID = 0)))
		MIhat = MI / max(H1,H2)

	return MIhat
