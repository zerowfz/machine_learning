#实现最简单的线性回归模型。

import numpy as np



def simple_1(X,Y):
    #X: 输入数据
    #Y:输入标签
    w = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),Y)
    return w


