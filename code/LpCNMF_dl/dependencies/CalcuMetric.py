import numpy as np
from dependencies.bestMap import bestMap
from dependencies.MutualInfo import MutualInfo
from collections import Counter
from dependencies.utils import LogiMul1d

def CalcuMetric(gndSet,label,semiSplit,supervise_flag):
    # Input: gndSet: groundtruth all label (nsample * 1)
    #        label:  predict all label (nsample * 1)
    #        semiSplit: True for labeled, Flase for unlabeled (nsample * 1)
    #        supervise_flag: selfsupervised: no labeled, all for test; semisupervised: only unlabeled for test
    if supervise_flag == 'selfsupervised':
        label = bestMap(gndSet.reshape(-1,1),label.reshape(-1,1))
        pur = Purity(gndSet, label)
        acc = ACC(gndSet, label)
        fscore = Fscore(gndSet,label)
        nmi = MutualInfo(gndSet,label)
    else:
        if supervise_flag == 'semisupervised':
            ungndSet = LogiMul1d(gndSet.reshape(-1,1), (~semiSplit).reshape(-1,1)) # 输出为列向量
            unlabel = LogiMul1d(label.reshape(-1,1), (~semiSplit).reshape(-1,1))
            prelabel = bestMap(ungndSet,unlabel)
            pur = Purity(ungndSet, prelabel)
            acc = ACC(ungndSet, prelabel)
            nmi = MutualInfo(ungndSet,prelabel)
            fscore = Fscore(ungndSet,prelabel)
    return pur, acc, fscore, nmi

################################################################
# Accuracy
################################################################
def ACC(y_true, y_pred):
    y_true = y_true.reshape(1,-1).copy()
    y_pred = y_pred.reshape(1,-1).copy()
    if y_true.shape[0] != y_pred.shape[0]:
        print('Error! The input size is not same!')
    acc = len(np.where(y_true == y_pred)[0]) / y_true.size
    return acc

################################################################
# Purity refer to https://en.wikipedia.org/wiki/Cluster_analysis
################################################################
def Purity(y_true, y_pred):
    # Input: y_true, y_pred are both row vectors 1 * nsample
    y_true = y_true.reshape(1,-1).copy()
    y_pred = y_pred.reshape(1,-1).copy()
    if y_true.shape[0] != y_pred.shape[0]:
        print('Error! The input size is not same!')
    num = y_true.size
    labels = np.unique(y_true) # row 
    labels_size = labels.shape[0]

    # the number of classes with the largest number of true classifications in each predicted class
    max_sum = np.zeros([labels_size])

    for i in range(labels_size):
        #  The position of each class in the true and predicted arrays
        idx = np.where(y_pred == labels[i])[1] # 找到列值
        if idx.size == 0:
            max_sum[i] = 0
        else:
            counter = Counter(y_true[0][idx])
            max_number = counter.most_common()[0][0]
            max_sum[i] = len(np.where(y_true[0][idx] == max_number)[0])

    pur = sum(max_sum)/num
    return pur

################################################################
# Fscore refer to https://en.wikipedia.org/wiki/F-score#cite_note-2
################################################################
def Fscore(P,C):
    #  P true class
    #  C predict class
    P = P.reshape(1,-1).copy()
    C = C.reshape(1,-1).copy() # 行向量直接输入，必须拉成行向量，不然Fscore里面一个矩阵计算会报错
    N = len(C) # sample number
    p = np.unique(P)
    c = np.unique(C)
    P_size = len(p) # number of true class
    C_size = len(c) #  number of predict class

    Pid = np.float64(np.dot(np.ones([P_size,1]),P) == np.dot(p.reshape(-1,1), np.ones([1,N])))
    Cid = np.float64(np.dot(np.ones([C_size,1]),C) == np.dot(c.reshape(-1,1), np.ones([1,N])))
    CP = np.dot(Cid, np.transpose(Pid)) # C*P
    Pj = np.sum(CP,0)
    Ci = np.sum(CP,1)
    
    precision = CP / (np.dot(Ci.reshape(-1,1), np.ones([1,P_size])))
    recall = CP / (np.dot(np.ones([C_size,1]), Pj.reshape(1,-1)))
    F = 2 * precision * recall / (precision + recall + 1e-6)
    #  total F
    FMeasure = sum((Pj/sum(Pj)) * np.max(F,0))
    return FMeasure