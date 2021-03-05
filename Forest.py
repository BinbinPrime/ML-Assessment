import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
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

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.9, random_state=1)

ForestR =[]
ForestModel = RandomForestClassifier(n_estimators=1000, min_samples_split=5, bootstrap=True)
ForestModel.fit(X_train, Y_train)
FRdata = ForestModel.predict(X_test)
FRreport = classification_report(Y_test, FRdata)
ACCper = metrics.accuracy_score(Y_test, FRdata)
ForestR.append(ACCper)
print(ForestR)
print(ACCper)
print(FRreport)



ForestR2 =[]
ForestModel2 = RandomForestClassifier(n_estimators=1000, min_samples_split=50, bootstrap=True)
ForestModel2.fit(X_train, Y_train)
FRdata2 = ForestModel2.predict(X_test)
FRreport2 = classification_report(Y_test, FRdata2)
ACCper2 = metrics.accuracy_score(Y_test, FRdata2)
ForestR2.append(ACCper2)
print(ForestR2)
print(ACCper2)
print(FRreport2)
