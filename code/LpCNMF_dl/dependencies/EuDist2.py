import numpy as np
import scipy

def EuDist2(fea_a,fea_b = np.array([2]),bSqrt = np.array([1])):
# %EUDIST2 Efficiently Compute the Euclidean Distance Matrix by Exploring the
# %Matlab matrix operations.
# %
# %   D = EuDist(fea_a,fea_b)
# %   fea_a:    nSample_a * nFeature
# %   fea_b:    nSample_b * nFeature
# %   D:      nSample_a * nSample_a
# %       or  nSample_a * nSample_b
# %
# %    Examples:
# %
# %       a = rand(500,10);
# %       b = rand(1000,10);
# %
# %       A = EuDist2(a); % A: 500*500
# %       D = EuDist2(a,b); % D: 500*1000
# %
# %   version 2.1 --November/2011
# %   version 2.0 --May/2009
# %   version 1.0 --November/2005
# %
# %   Written by Deng Cai (dengcai AT gmail.com)	
#     Translated by Yicheng Wang at 14th Oct 2021

	if fea_b.all() == np.array([2]).all():
		aa = np.sum(fea_a * fea_a,1)
		ab = np.dot(fea_a, np.transpose(fea_a))

		if type(aa) == scipy.sparse.coo.coo_matrix:
			aa = aa.todense()

		D = aa + np.transpose(aa) - 2*ab
		D[D < 0] = 0
		if bSqrt:
			D = np.sqrt(D)
		D = np.maximum(D, np.transpose(D))
	else:
		aa = np.sum(fea_a * fea_a,1)
		bb = np.sum(fea_b * fea_b,1)
		ab = np.dot(fea_a,np.transpose(fea_b))

		if type(aa) == scipy.sparse.coo.coo_matrix:
			aa = aa.todense()
			bb = bb.todense()

		aa = np.expand_dims(aa,axis = 1)
		bb = np.expand_dims(bb,axis = 1)
		D = aa + np.transpose(bb) - 2*ab
		D[D < 0] = 0
		if bSqrt:
			D = np.sqrt(D)
	return D