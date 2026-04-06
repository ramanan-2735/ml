import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix

data = load_breast_cancer()

df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model_id3 = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
model_id3.fit(X_train, y_train)

y_pred_id3 = model_id3.predict(X_test)

print("\nID3 (Entropy)")
print("Accuracy:", accuracy_score(y_test, y_pred_id3))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_id3))

model_cart = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
model_cart.fit(X_train, y_train)

y_pred_cart = model_cart.predict(X_test)

print("\nCART (Gini)")
print("Accuracy:", accuracy_score(y_test, y_pred_cart))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_cart))


plt.figure(figsize=(12, 6))
plot_tree(model_id3,
          feature_names=data.feature_names,
          class_names=data.target_names,
          filled=True)
plt.title("Decision Tree (ID3 - Entropy)")
plt.show()

plt.figure(figsize=(12, 6))
plot_tree(model_cart,
          feature_names=data.feature_names,
          class_names=data.target_names,
          filled=True)
plt.title("Decision Tree (CART - Gini)")
plt.show()