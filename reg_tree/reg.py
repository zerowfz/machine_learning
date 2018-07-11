#该脚本主要是回归树的一些基本函数
import numpy as np
import collections

def create_tree(dataset,leafType=leafType,errType=errType,opt=(1,4)):
    feat,val = chooseBestSplit(dataset,leafType,errType,opt)
    if feat == None:
        return val;
    retTree = collections.OrderedDict()
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet,rSet = binSplitDataSet(dataset,feat,val)
    retTree['left'] = create_tree(lSet,leafType,errType,opt)
    retTree['right'] = create_tree(rSet,leafType,errType,opt)
    return retTree

def chooseBestSplit(dataSet,leafType,errType,opt):
    tolS = opt[0]
    tolN = opt[1]
    if len(set(dataSet[:,-1])) == 1:
        return None,leafType(dataSet)
    n,d = dataSet.shape
    S = errType(dataSet)
    bestS = np.inf;bestIndex = 0;bestValue = 0
    for splitFea in range(d-1):
        for splitVal in set(dataSet[:,splitFea]):
            temdata1,temdata2 = binSplitDataSet(dataSet,splitFea,splitVal)
            if(temdata1.shape[0]<tolN) or (temdata2.shape[0]<tolN):
                continue;
            newS = errType(temdata1) + errType(temdata2)
            if newS < bestS:
                bestIndex = splitFea
                bestValue = splitVal
                bestS = newS
    if (S-bestS) < tolS:
        return None,leafType(dataSet)
    temdata1,temdata2 = binSplitDataSet(dataSet,splitFea,splitVal)
    if(temdata1.shape[0]<tolN) or (temdata2.shape[0]<tolN):
        return None,leafType(dataSet)
    return bestIndex,bestValue

def binSplitDataSet(dataset,fea,val):
    mask = dataset[:,fea]<=val
    return dataset[mask,:],dataset[~mask,:]

def leafType(dataSet):
    return np.mean(dataSet[:,-1])

def errType(dataSet):
    return np.var(dataSet[:,-1])*dataSet.shape[0]
'''        
#test for binSplit:
data = np.random.rand(4,5)
data1 = binSplitDataSet(data,0,0.5)
print(data)
print(data1)
'''
