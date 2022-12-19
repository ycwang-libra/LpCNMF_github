# -*- coding: utf-8 -*-
"""
all utils
"""
import numpy as np
import math
import torch
from scipy import sparse

def Config_Adjust(config):
    if config.dataset == 'MNIST':
        config.nEach = 500
        config.alpha = 10
        config.img_size = [28,28]
        config.end_dim = 16
        config.mFea = 784
    elif config.dataset == 'AR':
        config.nEach = 7 
        config.alpha = 1
        config.img_size = [165,120]
        config.end_dim = 315
        config.mFea = 19800
    elif config.dataset == 'COIL20':
        config.nEach = 72 
        config.percent = 0.2
        config.t = 10
        config.alpha = 1
        config.img_size = [32,32]
        config.end_dim = 16
        config.mFea = 1024
    elif config.dataset == 'COIL100':
        config.nEach = 72
        config.alpha = 1
        config.img_size = [32,32]
        config.end_dim = 16
        config.mFea = 1024
    elif config.dataset == 'UMIST':
        config.nEach = 19
        config.t = 0.1
        config.alpha = 10
        config.img_size = [28,23]
        config.end_dim = 12
        config.mFea = 644
    elif config.dataset == 'YaleB':
        config.nEach = 59
        config.t = 0.1
        config.alpha = 1000
        config.img_size = [32,32]
        config.end_dim = 16
        config.mFea = 1024
    elif config.dataset == 'Yale':
        config.nEach = 11
        config.t = 0.1
        config.alpha = 1000
        config.img_size = [32,32]
        config.end_dim = 16
        config.mFea = 1024
    elif config.dataset == 'USPS':
        config.nEach = 708
        config.alpha = 10
        config.img_size = [16,16]
        config.end_dim = 4
        config.mFea = 256
    return config

def NormalizeUV(U,Z,F,NormF,Norm):
    if Norm == 2:
        if NormF:
            norms = Np_max(1e-15,np.sqrt(np.sum(F**2,0)))
            F = np.array(np.dot(F,sparse.spdiags(norms**(-1),0,np.shape(F)[1],np.shape(F)[1]).todense()))
            normsu = Np_max(1e-15, np.sqrt(np.sum(U**2,0)))
            U = np.array(np.dot(U,sparse.spdiags(normsu**(-1),0,np.shape(U)[1],np.shape(U)[1]).todense()))
            Z_1 = sparse.spdiags(np.sqrt(norms),0,np.shape(F)[1],np.shape(F)[1]).todense()
            Z_2 = sparse.spdiags(np.sqrt(normsu),0,np.shape(Z)[1],np.shape(Z)[1]).todense()
            Z = np.array(np.dot(np.dot(Z_1,Z),Z_2))
    else:
        if NormF:
            norms = Np_max(1e-15,np.sum(np.abs(F),0))
            F = np.array(np.dot(F,sparse.spdiags(norms**(-1),0,np.shape(F)[1],np.shape(F)[1]).todense()))
            U = np.array(np.dot(U,sparse.spdiags(norms**(-1),0,np.shape(U)[1],np.shape(U)[1]).todense()))
            Z = np.array(np.dot(Z,sparse.spdiags(norms**(-1),0,np.shape(Z)[1],np.shape(Z)[1]).todense()))
    
    return U, Z, F

def Np_max(value,array):
    if len(array.shape)==1:
        N = array.shape[0] # column array 
        for i in range(N):
            array[i] = value if value > array[i] else array[i]
        return array
    elif len(array.shape)==2:
        N,M = array.shape  # 2d array
        for i in range(N):
            for j in range(M):
                array[i,j] = value if value > array[i,j] else array[i,j]
        return array

def Normalize(fea): # max min norm  put elements to (0~1)
    nSmp = fea.shape[0]
    nDim = fea.shape[1]
    # Feature Nomalization
    for i in range(nSmp):
        min = np.min(fea[i,:])
        max = np.max(fea[i,:])
        down = max - min
        for j in range(nDim):
            fea[i][j] = (fea[i][j] - min) / down
    return fea

def NormalizeFea(fea, device): # L2-norm sum(xi^2) = 1
    nSmp = fea.shape[0]
    nDim = fea.shape[1]
    # Feature Nomalization
    if device == torch.device('cpu'):
        feaNorm = Twoarraymax(np.ones(nSmp)*1e-14,np.sum(fea ** 2, 1))
        feaNorm = 1/np.sqrt(feaNorm)
        for i in range(nSmp):
            for j in range(nDim):
                fea[i][j] = feaNorm[i] * fea[i][j]
        return fea
    else:
        fea_cuda = torch.from_numpy(fea).to(device)
        feaNorm_cuda = Twotensormax(torch.ones(nSmp).to(device)*1e-14,
                                    torch.sum(fea_cuda**2,1),device)
        feaNorm_cuda = 1/torch.sqrt(feaNorm_cuda)
        for i in range(nSmp):
            for j in range(nDim):
                fea[i][j] = feaNorm_cuda[i] * fea_cuda[i][j]
        return fea

def Normalize255(InImg):
# 	norm to 0~255
	ymax = 255
	ymin = 0
	xmax = np.max(InImg)
	xmin = np.min(InImg)
	Out = np.round((ymax-ymin)*(InImg-xmin)/(xmax-xmin) + ymin)
	return Out

def Twoarraymax(A,B):
    # compare array each element one by one to get the max
    if A.shape != B.shape:
        print('Two np.array must be the same size!')
        exit()
    else:
        N = A.shape[0]
        result = np.zeros(N)
        for i in range(N):
            if A[i] >= B[i]:
                result[i] = A[i]
            else:
                result[i] = B[i]
    return result

def Twotensormax(A,B,device):
    # compare tensor each element one by one to get the max
    if A.shape != B.shape:
        print('Two torch.tensor must be the same size!')
        exit()
    else:
        N = A.shape[0]
        result = torch.zeros(N).to(device)
        for i in range(N):
            if A[i] >= B[i]:
                result[i] = A[i]
            else:
                result[i] = B[i]
    return result

def CalculateObj(X, U, Z, F, S, A, L):
    MAXARRAY = 5000*1024*1024/8 
    # % 500M. You can modify this number based on your machine's computational power.
    deltaVU = 0
    dVordU = 1
    
    dV = []
    nSmp = X.shape[1]
    mn = X.shape[0]*X.shape[1]
    nBlock = math.ceil(mn/MAXARRAY)
    V = np.dot(F,Z)
    
    if mn < MAXARRAY:
        dX = X - np.dot(U,np.transpose(V))
        obj_NMF = np.sum(dX ** 2)
        if deltaVU:
            if dVordU:
                dV = np.dot(np.transpose(dX),U)+np.dot(L,V)
            else:
                dV = np.dot(dX,V)
    else:
        obj_NMF = 0

        PatchSize = math.ceil(nSmp/nBlock)
        for i in range(nBlock):
            if (i+1)*PatchSize > nSmp:
                smpIdx = [x+i*PatchSize+1 for x in range(nSmp-i*PatchSize)]
            else:
                smpIdx = [x+i*PatchSize+1 for x in range((i+1)*PatchSize-i*PatchSize)]
            dX = np.dot(U,np.transpose(V[smpIdx-1,:]))-X[:,smpIdx-1]
            obj_NMF = obj_NMF + np.sum(dX ** 2)

            dV[smpIdx-1,:] = np.dot(np.transpose(dX),U)
            
    sum1 = np.sum(np.dot(np.transpose(F),L) * np.transpose(F))
    FA = F - A
    sum2 = np.sum(np.dot(np.transpose(FA),S) * np.transpose(FA))
    obj_Lap = sum1 + sum2
    
    obj = obj_NMF + obj_Lap
    return obj

def torch_CalculateObj(X, U, Z, F, S, A, L):
    MAXARRAY = 5000*1024*1024/8 
    # % 500M. You can modify this number based on your machine's computational power.
    deltaVU = 0
    dVordU = 1
    
    dV = []
    nSmp = X.shape[1]
    mn = X.shape[0]*X.shape[1]
    nBlock = math.ceil(mn/MAXARRAY)
    V = torch.mm(F,Z)
    
    if mn < MAXARRAY:
        dX = X - torch.mm(U,V.permute(1,0))
        obj_NMF = torch.sum(dX ** 2)
        if deltaVU:
            if dVordU:
                dV = torch.mm(dX.permute(1,0),U)+torch.mm(L,V)
            else:
                dV = torch.mm(dX,V)
    else:
        obj_NMF = 0

        PatchSize = math.ceil(nSmp/nBlock)
        for i in range(nBlock):
            if (i+1)*PatchSize > nSmp:
                smpIdx = [x+i*PatchSize+1 for x in range(nSmp-i*PatchSize)]
            else:
                smpIdx = [x+i*PatchSize+1 for x in range((i+1)*PatchSize-i*PatchSize)]
            dX = torch.mm(U,(V[smpIdx-1,:]).permute(1,0))-X[:,smpIdx-1]
            obj_NMF = obj_NMF + torch.sum(dX ** 2)

            dV[smpIdx-1,:] = torch.mm(dX.permute(1,0),U)
            
    sum1 = torch.sum(torch.mm(F.permute(1,0),L) * F.permute(1,0))
    FA = F - A
    sum2 = torch.sum(torch.mm((FA).permute(1,0),S) * (FA).permute(1,0))
    obj_Lap = sum1 + sum2
    
    obj = obj_NMF + obj_Lap
    return obj

def LogiMul1d(A,Logi):
# =============================================================================
# according the location of 按照Logi中0的位置，去掉A中对应行
#     Input:  A: 1xd array
#             Logi: dx1 array and Logi is 0 or 1 logical array
#     Output: A * Logi without Logi=0
# =============================================================================
    idx = np.where(Logi == 0)[0]
    Result = np.delete(A,idx,0) #删掉idx行
    return Result

def LogiMul2d(A,Logi):
# =============================================================================
# 按照Logi中0的位置，去掉A中对应元素，再按照列拉成向量
#     Input:  A: mxn array
#             Logi: mxn array and Logi is 0 or 1 logical array
#     Output: A * Logi as 1d array without Logi=0
# =============================================================================
    if A.shape != Logi.shape:
        print('The two input musr be the same size!')
    else:
        row_idx = np.where(Logi != 0)[0]
        col_idx = np.where(Logi != 0)[1]
        num_Nozero = len(row_idx)
        Result = np.zeros([num_Nozero])
        for i in range(num_Nozero):
            Result[i] = A[row_idx[i], col_idx[i]]
        return Result

def LogiReplace(A,Logi,B,ID):
# =============================================================================
# A中元素按照Logi对应行或列位置依次替换为B中元素
#     Input:  A: kxd array Oral data
#             Logi: dx1 array and Logi is 0 or 1 logical array
#             B: 1xs array (s<d and s = the number of the No zero in Logi)
#     Output: 1xd array Using elements of B to replace A according to Logi
# =============================================================================
    idx = np.where(Logi != 0)[0]    
    num_Nozero = len(idx)
    Result = A.copy()
    # print('A shape is :'+ str(A.shape))
    # print('Logi shape is :'+ str(Logi.shape))
    if ID == 'row': # 按照行方向把A中对应某列全换为B的值或者对应列
        if A.shape[1] != Logi.shape[0]:
            print('Oral data and Logi must be the same size!')
        else:
            if type(B) == np.int64: # B为数值类型
                for i in range(num_Nozero):
                    Result[:,idx[i]] = B
            else:                                  # B为向量
                if num_Nozero != B.shape[1]:
                    print('aimed data must be the same size as the number of the No zero in Logi!')
                else:
                    for i in range(num_Nozero):
                        Result[:,idx[i]] = B[:,i]
    elif ID == 'col':
        if A.shape[0] != Logi.shape[0]:
            print('Oral data and Logi must be the same size!')
        else:
            if type(B) == np.int64: # B为数值类型
                for i in range(num_Nozero):
                    Result[idx[i]] = B
            else:                                  # B为向量
                if num_Nozero != B.shape[1]:
                    print('aimed data must be the same size as the number of the No zero in Logi!')
                else:
                    for i in range(num_Nozero):
                        Result[idx[i]] = B[:,i]
    return Result

def MATLAB_Mat(Mat, array, value):
# =============================================================================
# 矩阵元素替换，矩阵按照列拉成向量，对应array作为指标，指向元素替换为value
#     Describe: replace the element of Matrix in idx of array with aim value
#               as the MATLAB does
#     Input: Mat: (numpy array) mxn  matrix
#            array: (numpy array) idx of the element need to be replace
#            value: the aim value
#     Example: (in matlab) a=[1,2,3;4,5,6;7,8,9]; a(2) is 4; a(5) is 5; a(6) is 8
# =============================================================================
    M, N = Mat.shape
    num = len(array)
    if num > M * N:
        print('The array is too large!')
        exit()
    else:
        for i in range(num):
            row = array[i] % M
            column = np.int64(np.floor(array[i]/M))
            Mat[row,column] = value
        return Mat
        
def MATLAB233(array,n,k,value):
# =============================================================================
# litekmeans233行sparse,生成n x k全零矩阵，这个全零矩阵中按照行，array中每个值对应的
# 列该元素替换为value
#     Describe: replace the element of all zero Matrix in idx of array with aim 
#               value as the MATLAB does
#     Input: 
#            array: (1d-numpy array) idx of the element need to be replace
#            n, k : the size of all zero matrix
#            value: the aim value
#     Output:
#            matrix: n x k zeros matrix with corresponding elements replaced by 
#            value
# =============================================================================
    if n < len(array): # 矩阵行数比array短 pass 
        exit()
    elif max(array)+1>k: # 矩阵列数比array中指标小 pass 
        exit()
    else:
        Result = np.zeros([n,k],dtype = 'uint8')
        for i in range(n):
            Result[i, array[i]] = value
    return Result
    
def LogiMerge(Logi1, Logi2, relation):
# =============================================================================
# 两个相同尺寸逻辑列向量(n x 1)按照逻辑关系合并为一个逻辑列向量
#     Describe: Merge two Logi array according relationship
#     Input: 
#            Logi1 and Logi2 are n x 1 Logic array
#            relation str 'and' or 'or'
#     Output:
#            LOGI are n x 1 Logic array
# =============================================================================
    Logi1 = Logi1.reshape(-1,1)
    Logi2 = Logi2.reshape(-1,1)
    if Logi1.shape != Logi2.shape:
        print('The two logi array must be the same size!')
    else:
        N = Logi1.shape[0]
        LOGI = np.zeros([N,1],dtype = 'bool')
        if relation == 'and':
            for i in range(N):
                LOGI[i,0] = Logi1[i,0] and Logi2[i,0]
        elif relation == 'or':
            for i in range(N):
                LOGI[i,0] = Logi1[i,0] or Logi2[i,0]
        return LOGI

def ReplaceDiag(Matrix, value):
# =============================================================================
# 方阵的对角元素替换为value
# =============================================================================
    m, n = Matrix.shape
    if m != n:
        print('Input matrix must be square matrix!')
    else:
        for i in range(m):
            Matrix[i,i] = value
    return Matrix

def Concat_Array322(array1, array2):
# =============================================================================
# matlab hungarian322行U中指标的拼接，[n+1, empty] 当empty为空时输出[n+1]，不为空时拼接上
# =============================================================================
    if type(array1) == int: # 第一个是数值
        if type(array2) == int:  # 第一个是数值，第二个也是数值
            return np.array([array1, array2])
        elif type(array2) == np.ndarray:
            if array2.size == 0: # 第一个是数值，第二个是空array
                return np.array([array1])
            else:               # 第一个是数值，第二个是实array
                return np.append(np.array([array1]),array2)
        else:                    # 第一个是数值，第二个格式未知
            print('The first array is int, the second must be a int or array!')
    elif type(array1) == np.ndarray:  # 第一个是array
        if array1.size == 0:   # 第一个是空array
            if type(array2) == int: # 第一个是空array,第二个是数值
                return np.array([array2])
            elif type(array2) == np.ndarray:  # 第二个是array
                if array2.size == 0:  # 第一个是空array,第二个是空array
                    print('The first array is empty, the second must be a int or real array!')
                else:                 # 第一个是空array,第二个是实array
                    return array2  # 实array2
            else:               # 第一个是空array ，第二个格式未知
                print('The first array is empty, the second must be a int or real array!')
        else:  # 第一个是实array
            if type(array2) == np.ndarray:
                if array2.size == 0: # 第一个是实array，第二个是空array
                    return array1
                else:                # 第一个是实array，第二个是实array
                    return np.append(array1,array2)
            elif type(array2) == int: # 第一个是实array，第二个是数值
                return np.append(array1,np.array([array2]))
            else:                # 第一个是实array ，第二个格式未知
                print('The first array is real array, the second must be a int or array!')
    else:                      # 第一个格式未知
        print('The first input must be a int or array!')

def MATLAB_sparse(array1,array2,value):
# =============================================================================
# matlab 稀疏矩阵构造，按照array1和array2作为横纵坐标，对应稀疏矩阵元素填入value
# =============================================================================
    if array1.shape != array2.shape:
        print('The two axises must be the same size!')
    else:
        N = array1.shape[0]
        Result = np.zeros([N,N])
        for i in range(N):
            Result[array1[i]-1,array2[i]-1] = value
        return Result

def Mat2Array(Mat, ID):
# =============================================================================
# 二维矩阵拉成一维（ID = 0 按列（相当于matlab中A(:)，ID = 1 按行）
# =============================================================================
    M,N = Mat.shape
    Result = np.zeros([M*N])
    t = 0
    if ID == 1:
        for i in range(M):
            for j in range(N):
                Result[t] = Mat[i,j]
                t = t + 1
    elif ID == 0:
        for j in range(N):
            for i in range(M):
                Result[t] = Mat[i,j]
                t = t + 1
    else:
        print('The input ID is invalid!')
    return Result

def MATLAB_hungarian187(A,k,n,cols):
# =============================================================================
# malab hungarian函数187行实现  A[k,[n,cols]] = [-(cols+1),0] 拼接脚标 并替换
# cols 值为python指标，从0开始
# =============================================================================
    if len(cols)==1: # cols为单值
        A[k,[n,cols[0]]] = [-(cols[0]+1),0]
    elif len(cols)>1:  # cols为向量
        idx = np.hstack((np.array([n],dtype='int32'),cols))
        A[k,idx] = np.hstack((-(cols+1), np.array([0],dtype='int32')))
    return A

def Mat_hungarian415(A, x, y):
# =============================================================================
# malab hungarian函数415行实现  提取矩阵A在行列x,y中的值
# =============================================================================
    m = len(x)
    n = len(y)
    Result = np.zeros([m,n])
    for i in range(m):
        for j in range(n):
            Result[i,j] = A[x[i],y[j]]
    return Result

def Mat_hungarian418(A, x, y, value, ID):
# =============================================================================
# malab hungarian函数415行实现  对矩阵A在行列x,y中的值 减去某值
# =============================================================================
    m,n = A.shape
    Result = A.copy()
    if ID == 'subtract':
        for i in range(len(x)):
            for j in range(len(y)):
                Result[x[i],y[j]] = A[x[i],y[j]] - value
    elif ID == 'add':
        for i in range(len(x)):
            for j in range(len(y)):
                Result[x[i],y[j]] = A[x[i],y[j]] + value
    return Result

