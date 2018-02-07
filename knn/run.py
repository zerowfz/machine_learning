import data
import knn
import numpy as np
k=1
def dis_square(X1,X2):
    num_X1 = X1.shape[0]
    return np.sum((X1-np.tile(X2,(num_X1,1)))**2,1)
[Xtrain,Ytrain,Xtest,Ytest] = data.get_data()
num_test = Xtest.shape[0]
pred = np.zeros(num_test)
for i in range(num_test):
    pred[i] = knn.knn_base(Xtrain,Ytrain,Xtest[i,:],k,dis_square)
pre = sum(pred==Ytest)/num_test
print(pre)

