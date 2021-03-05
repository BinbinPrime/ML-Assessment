import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Import Data
df = pd.read_csv("D:\pythonProject2\clinical_dataset.csv")

# Draw Plot

sns.kdeplot(df.loc[df['Status'] == 'healthy', "BMI"], shade=True, color="g", label="healthy", alpha=.7)
sns.kdeplot(df.loc[df['Status'] == 'cancerous', "BMI"], shade=True, color="black", label="cancerous", alpha=.7)

# Decoration
plt.title('Density Plot of Status and BMI', fontsize=20)
plt.xlabel("BMI")
plt.ylabel("Density")
plt.legend()
plt.show()