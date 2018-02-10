import numpy as np
import copy
'''
本脚本主要用来实现决策树的相关内容。
constrcut_tree:该函数是构建决策树的主要函数
其输入：数据集X:n*p n:样本数，p-1维特征，p为样本类别,
以及属性信息label:属性名称,p-1一维数组，label表示的是此时X每一列对应的属性名称
决策结构用字典来表示，例如{attribution1:{0:{attribution2:{}},1:{attribution3:{}}}
'''

def construct_tree(X,label):
    
    classList = [sample[-1] for sample in X]
    #如果此时所有的样本的类别相同，返回该类别。
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    #如果此时对应属性已经划分完毕
    if len(X[0])==1:
        return return_major(classList)
    #如果此时划分之后的子集为空，但是显然这是不可能的，对于这种情况来说，
    #因为我们后面的编程过程中，我的属性划分的个数是根据，此时样本的属性数
    #得到的，而不是一开始默认的，注意于西瓜书上算法的区别

    #选择最优划分属性:
    bestFea = bestdived(X)
    bestFeaName = label[bestFea]
    feaValue = [x[bestFea] for x in X]
    uniqueValue = set(feaValue)
    myTree = {bestFeaName:{}}
    del(label[bestFea])
    for i in uniqueValue:
        myTree[bestFeaName][i]=construct_tree(splitDataSet(X,bestFea,i),label)
    return myTree




#统计一组数据中，出现次数最多的时候用以下代码
def return_major(Y):
    #给定一组类别，返回这组数据中，最大的类别
    label_count={}
    for i in Y:
        label_count[i] = label_count.get(i,0)+1
    sorted_class = sorted(label_count.items(),key=operator.itemgetter(1),reverse=True)
    return sorted_class[0][0]

def splitDataSet(X,fea,value):
    #根据属性的某个值得到相应的数据集
    y = []
    tem = copy.deepcopy(X)
    for i in tem:
        if i[fea] == value:
            del(i[fea])
            y.append(i)
    return y

def bestdived(X):
    #对任何一个特征进行划分，计算得到的数据集的熵。然后计算
    #这个特征对应的信息增益
    baseEnt = calcEnt(X)
    tem0 = 0#记录最大的信息增益
    for i in range(len(X[0])-1):
        #fea 循环
        feaValue = [x[i] for x in X]
        uniqueValue = set(feaValue)
        tem1 = 0#记录该特征划分的子集熵的总和
        for j in uniqueValue:
            subDataset = splitDataSet(X,i,j)
            prob = len(subDataset)/len(X)
            tem1 = tem1 + prob*calcEnt(subDataset)
        infoGain = baseEnt - tem1
        if infoGain > tem0:
            tem0 = infoGain
            bestFea = i
    return bestFea

def calcEnt(X):
    #计算数据即X的熵，此时的熵是当对于类别信息来的。
    labelCount = {}
    for i in X:
        i = i[-1]
        labelCount[i] = labelCount.get(i,0)+1;
    tem = np.array(list(labelCount.values()))
    tem = tem/len(X)
    return np.sum(-np.log(tem)*tem)


