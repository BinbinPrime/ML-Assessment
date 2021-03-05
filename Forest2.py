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
for i in range(100,5000,500):
    ForestModel = RandomForestClassifier(n_estimators=i, min_samples_split=5, bootstrap=True)
    ForestModel.fit(X_train, Y_train)
    FRdata = ForestModel.predict(X_test)
    FRreport = classification_report(Y_test, FRdata)
    ACCper = metrics.accuracy_score(Y_test, FRdata)
    ForestR.append(ACCper)
    print(ACCper)
    print(FRreport)
    print(ForestR)
    continue

Range = (100, 600, 1100, 1600, 2100, 2600, 3100, 3600, 4100, 5000)
# Range = (100, 300, 500, 700, 900, 1100, 1300, 1500, 1700, 2000)
# Range = (100, 200, 300, 400, 500, 600, 700, 800, 900)
plt.plot(Range, ForestR, label='Accuracy', color='red')
plt.xlabel("Tree Number")
plt.ylabel("Accuracy")
plt.legend()
plt.show()