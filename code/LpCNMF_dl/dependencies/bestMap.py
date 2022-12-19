# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 11:31:06 2021

@author: yc_wang
"""

import numpy as np
from dependencies.utils import LogiMerge, LogiReplace
from dependencies.hungarian import hungarian

def bestMap(L1, L2):
# %bestmap: permute labels of L2 to match L1 as good as possible
# %   [newL2] = bestMap(L1,L2);
# %
# %   version 2.0 --May/2007
# %   version 1.0 --November/2003
# %
# %   Written by Deng Cai (dengcai AT gmail.com)
# %   Translated by Yicheng Wang
	if L1.shape != L2.shape:
		print('size(L1) must == size(L2)!')
		exit()

	Label1 = np.unique(L1)
	nClass1 = len(Label1)
	Label2 = np.unique(L2)
	nClass2 = len(Label2)

	nClass = max(nClass1, nClass2)
	G = np.zeros([nClass, nClass], dtype = 'int64')
	for i in range(nClass1):
		for j in range(nClass2):
			G[i,j] = len(np.where(LogiMerge(L1==Label1[i],L2==Label2[j],'and')==True)[0])

	c, t = hungarian(-G) # hungarian done debugged 与matlab一致
	newL2 = np.zeros([L2.shape[0]])
	for i in range(nClass2):
		newL2 = LogiReplace(newL2, L2 == Label2[i], Label1[int(c[i]-1)],'col')
	return newL2

