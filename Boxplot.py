import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('D:\pythonProject2\clinical_dataset.csv',usecols=["Age"])
data = np.array(data)
healthy = data[1:64:1]
cancerous = data[64:128:1]
# Mean = print(healthy.mean(axis=0))

pic = {
'Health': [52, 41, 40, 27, 32, 55, 46, 40, 28, 32,53,48,83,82,68,86,49,89,76,73,75,34,29,25,24,38,44,47,61,64,32,36,34,29,35,54,45,50,66,35,36,66,53,28,43,51,67,66,69,60,77,76,76,75,69,71,66,75,78,69,85,76,77,55,55],
'Cancerous': [45, 45, 49, 34, 42, 68, 51, 62, 38, 69,69,49,51,59,45,54,64,46,44,45,44,51,72,46,43,55,43,86,41,59,81,48,71,42,65,48,85,48,58,40,82,52,49,60,49,44,40,71,69,74,66,65,72,57,73,45,46,68,75,54,45,62,65,72,86],
}

df = pd.DataFrame(pic)
df.plot.box(title="Boxplot")
df.boxplot()
plt.xlabel("Status")
plt.ylabel("Age")
plt.show()





