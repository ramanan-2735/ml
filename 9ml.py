from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
data = load_breast_cancer()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
ada = AdaBoostClassifier(n_estimators=100, random_state=42)
ada.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_pred_ada = ada.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
acc_ada = accuracy_score(y_test, y_pred_ada)
print("Model Performance:\n")
print(f"Random Forest Accuracy: {acc_rf:.4f}")
print(f"AdaBoost Accuracy: {acc_ada:.4f}")
models = ['Random Forest', 'AdaBoost']
accuracies = [acc_rf, acc_ada]
plt.figure()
plt.bar(models, accuracies)
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Random Forest vs AdaBoost Performance")
plt.show()