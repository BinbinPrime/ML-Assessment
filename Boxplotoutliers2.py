#-*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt

clinical_dataset = 'D:\pythonProject2\clinical_dataset.xlsx'
data = pd.read_excel(clinical_dataset, usecols=["Age","MCP.1"],index_col = u'Age')


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.figure()
p = data.boxplot()
plt.show()