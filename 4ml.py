import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score

data = load_digits()

X = data.data
y = data.target

print("Dataset shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


k_values = [1, 3, 5, 7, 9]
precision_list = []
recall_list = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    precision_list.append(precision)
    recall_list.append(recall)
    
    print(f"\nK = {k}")
    print("Precision:", precision)
    print("Recall:", recall)


plt.figure()
plt.plot(k_values, precision_list, marker='o', label='Precision')
plt.plot(k_values, recall_list, marker='o', label='Recall')

plt.xlabel("Value of K")
plt.ylabel("Score")
plt.title("K vs Precision & Recall")
plt.legend()
plt.show()