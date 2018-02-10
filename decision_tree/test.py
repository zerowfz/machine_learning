import base
import classify
dataSet = [[1,1,'yes'],
           [1,1,'yes'],
           [1,0,'no'] ,
           [0,1,'no'],
           [0,1,'no']]

label = ['fea1','fea2']
tree = base.construct_tree(dataSet,label)
label = ['fea1','fea2']
print(classify.classify(tree,label,[0,0]))
