import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

iris = load_iris()
df = pd.DataFrame(data=iris.data,columns=iris.feature_names)
df['species'] = iris.target

plt.figure()
plt.scatter(df['sepal length (cm)'],df['sepal width (cm)'])
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.title("Length or width")
plt.show()

scaler = MinMaxScaler()
df_scaled = df.copy()
df_scaled[df.columns[:-1]] = scaler.fit_transform(df[df.columns[:-1]])
df_scaled.hist(figsize=(10,6))
plt.show()

sns.pairplot(df,hue="species")
plt.show()
sns.boxplot(data=df.iloc[:, :-1])
plt.show()




