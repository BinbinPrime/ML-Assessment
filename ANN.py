#-*- coding: utf-8 -*-
from cProfile import label
from random import random
from turtle import color
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier


dataframe = pd.read_excel('D:\pythonProject2\clinical_dataset.xlsx')
data = dataframe.columns.values.tolist()
df = dataframe[data[0:9]]
df = (df - df.min()) / (df.max() - df.min())

data2 = df.values
X = data2[:, 0:9]
y = dataframe['Status']
Y = []
for i in y:
    if (i == "healthy"):
        Y.append(0)
    else:
        Y.append(1)
Y = np.array(Y)

# ANNModel = MLPClassifier(hidden_layer_sizes=[500, 500], activation='logistic', solver='lbfgs', random_state=0
#                          ,alpha=0.1,max_iter=100)
# ANNModel.fit(X_train, Y_train)
# ACC = ANNModel.score(X_test,Y_test)
# ACCreport.append(ACC)
# print('Accuracy is:{:.3f}'.format(ACC))
# print(ACCreport)

# Y_pred = ANNModel.predict(X_test)
# ACCreport = classification_report(Y_test, Y_pred)
# print(ACCreport)
# ACCreport2 = np.array(ACCreport)

# epochs = [20,30,50,80,100,120,140,160,180,200]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.9)
ACCreport =[]
for i in range(20,200,20):
    ANNModel = MLPClassifier(hidden_layer_sizes=[500, 500], activation='logistic', solver='lbfgs', random_state=0,
                             alpha=0.1, max_iter = i)
    ANNModel.fit(X_train, Y_train)
    ACC = ANNModel.score(X_test, Y_test)
    ACCreport.append(ACC)
    print('Accuracy is:{:.3f}'.format(ACC))
    print(ACCreport)


# accuracy = [0.692,0.769,0.692,0.654,0.846,0.796,0.846,0.846,0.846,0.846]
# epochs = [20,40,60,80,100,120,140,160,180,200]
# plt.plot(epochs,accuracy,label='accuracy',color='blue')
# plt.plot(range(20,200,20),ACCreport, label='Accuracy',color='blue')
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.show()

hidden_layer = [50, 500, 1000]
cv = sklearn.model_selection.KFold(n_splits=10, shuffle=False, random_state=1)
scores = cross_val_score(hidden_layer,X_train,X_test,cv,scoring='accuracy')
print(scores)