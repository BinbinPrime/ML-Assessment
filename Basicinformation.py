import pandas as pd
import numpy as np
import operator
from functools import reduce
import matplotlib.pyplot as plt
from boto import sns
from pandas import DataFrame
from pandas.tests.test_downstream import df

data = pd.read_csv("D:\pythonProject2\clinical_dataset.csv")
data = np.array(data)

data2 = pd.read_csv("D:\pythonProject2\clinical_dataset.csv",usecols=["Age","BMI","Glucose","Insulin","HOMA","Leptin","Adiponectin","Resistin","MCP.1"])
data2 = np.array(data2)
# print(data2)


Max = print(data2.max(axis=0))

Min = print(data2.min(axis=0))

Mean = print(data2.mean(axis=0))

Std = print(data2.std(axis=0))

print(data2.shape)

print(np.isnan(data2).any())


data3 = pd.read_csv('D:\pythonProject2\clinical_dataset.csv',usecols=["Age"])
data3 = np.array(data3)

data4 = pd.read_csv('D:\pythonProject2\clinical_dataset.csv',usecols=["Status"])
data4 = np.array(data4)

data5 = pd.read_csv('D:\pythonProject2\clinical_dataset.csv',usecols=["MCP.1"])
data5 = np.array(data5)

res = data4[1:63:1]
res3 = np.zeros((64,), dtype=int)
res2 = data4[64:128:1]
res4 = np.ones((64,), dtype=int)


# pic = {'healthy': data3,
# 'cancerous': data4
# }
# df = pd.DataFrame(pic)
# df.plot.box(title="Consumer spending in each country")
# plt.grid(linestyle="--", alpha=0.3)
# plt.show()


# BMIHealthy = df[df['data4'] == 'healthy']['BMI']
# BMICancerous = df[df['data4'] == 'cancerous']['BMI']
# sns.kdeplot(BMIHealthy)
# sns.kdeplot(BMICancerous)
# plt.title('1')
# plt.legend()
# plt.show()

# pic = {
# 'Health': [52, 41, 40, 27, 32, 55, 46, 40, 28, 32],
# 'Cancerous': [45, 45, 49, 34, 42, 68, 51, 62, 38, 69],
# }
# df = pd.DataFrame(pic)
# df.plot.box(title="Consumer spending in each country")
# plt.grid(linestyle="--", alpha=0.3)
# plt.show()

