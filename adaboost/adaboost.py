import numpy as np


def loadSimpleData():
    dataMat = np.array([[1.,2.1],
        [2.,1.1],
        [1.3,1.],
        [1.,1.],
        [2.,1.]])
    classLabels = np.array([1.0,1.0,-1.0,-1.0,1.0])
    return dataMat,classLabels

def classify(data,dim,p,threshold):
    if data.size == data.shape[0]:
        predict = np.ones(1)
    else:
        predict = np.ones(data.shape[0])
    if p=='le':
        if(data.size==data.shape[0]):
            predict[data[dim]>threshold] = -1
        else:
            predict[(data[:,dim]>threshold)] = -1
    else:
        if (data.size==data.shape[0]):
            predict[data[dim]<threshold] = -1
        else:
            predict[(data[:,dim]<threshold)] = -1
    return predict

#D：每一个样本对用的权重
def build_tree_base(data,label,D):
    n,d = data.shape
    bestC = {}
    min_error = 10000 
    num_step = 10 #对于连续特征的划分间隔
    step_size = (np.max(data,0)-np.min(data,0))/num_step
    for i in range(d):
        for j in range(num_step):
            threshold = np.min(data,0)[i] + j*step_size[i] 
            for p in ['le','gt']:
                #分类：
                predict = classify(data,i,p,threshold)
                error = np.ones(n)
                error[predict==label]=0
                weight_error = np.dot(error,D)
                #print( " split :dim %d, threshold %.2f,ineq %s ,the weighted error %.3f"%(i,threshold,p,weight_error))
                if weight_error<min_error:
                    min_error = weight_error
                    predict_best = predict
                    bestC['dim'] = i
                    bestC['threshold'] = threshold
                    bestC['ineq'] = p
    return bestC,min_error,predict_best

def adaboost(data,label,maxIter):
    weakClassArr = []
    n,d = data.shape
    D = np.ones(n)/n
    aggPredict = np.zeros(n)
    for i in range(maxIter):
        base,error,predict = build_tree_base(data,label,D)
        alpha = np.log((1-error)/error)/2
        base['alpha'] = alpha
        weakClassArr.append(base)
        tem = np.ones(n)
        tem[predict==label]=-1
        D = D*np.exp(tem*alpha)
        D = D/D.sum()
        aggPredict += alpha*predict
        #print("aggpredict",aggPredict)
        aggError = np.dot(np.sign(aggPredict)!=label,np.ones(n))
        errorRate = aggError/n
        #print("errorRate:",errorRate)
        if errorRate==0:
            break
    return weakClassArr

def adaclassify(data,classifierArr):
    if data.size == data.shape[0]:
        aggPredict = np.zeros(1)
    else:
        aggPredict = np.zeros(data.shape[0])
    for i in classifierArr:
        predict = classify(data,i['dim'],i['ineq'],i['threshold'])
        print(predict)
        aggPredict += i['alpha']*predict
    return aggPredict





###测试程序
data,label = loadSimpleData()
c = adaboost(data,label,40)
print(c)
predict = adaclassify(2*np.ones(2),c)
print(predict)
def test_for_adaboost():
    data,label = loadSimpleData()
    c=adaboost(data,label,40)
    print(c)
def test_for_build_base():
    data,label = loadSimpleData()
    x=build_tree_base(data,label,np.ones(data.shape[0])/data.shape[0])
    print(x)
