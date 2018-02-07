#主要实现knn算法
#knn算法 输入：
#X,Y输入数据(r:n,c:feadim)以及标签,K：knn中的k，dis：knn中的距离度量
#Xtest:一个输入数据
import numpy as np
import operator
def knn_base(Xtrain,Ytrain,Xtest,k,dis):
    #计算测试样本和所有元样本之间的距离，返回的是n*c
	#每一列表示的是一个测试样本和原始样本之间的距离
    distance = dist(Xtrain,Xtest)
    sort_index = distance.argsort()
	vote = {}
	for i in range(k):
	    label = Ytrain[sort_index[i]]
		vote[lable] = vote.get(label,0)+1
	label_count = sorted(vote.items(),key=operator.itemgetter(1),reverse=True);
	return label_count[0][0]

def dis_square(X1,X2):
    num_X1 = X1.shape[1]
    return np.sum((X1-np.tile(X2,(num_X1,1))) **2,0)   
    
