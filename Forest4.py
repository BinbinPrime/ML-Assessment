import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score

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

hidden_layer = [50, 500, 1000]
X2 = np.array(X)
Y2 = np.array(Y)
for i in hidden_layer:
    ANNModel = MLPClassifier(hidden_layer_sizes=[i, i], activation='logistic', solver='lbfgs', random_state=1,
                         alpha=0.1, max_iter=200)
    cv = sklearn.model_selection.KFold(n_splits=10, shuffle=True, random_state=1)
    scores = cross_val_score(ANNModel, X2, Y2, cv=cv)
    print("Accuracy: %0.2f (+/- %0.2f)" % (i, scores.mean()))

tree_num = [20, 500, 10000]
X3 = np.array(X)
Y3 = np.array(Y)
for j in tree_num:
   ForestModel = RandomForestClassifier(n_estimators=j, min_samples_split=5, bootstrap=True)
   cv = sklearn.model_selection.KFold(n_splits=10, shuffle=True, random_state=1)
   scores2 = cross_val_score(ForestModel, X3, Y3, cv=cv)
   print("Accuracy: %0.2f (+/- %0.2f)" % (j, scores2.mean()))

