from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

manual_model = LogisticRegression(max_iter=200)
manual_model.fit(X_train, y_train)

y_pred_manual = manual_model.predict(X_test)
acc_manual = accuracy_score(y_test, y_pred_manual)

print("Manual Model Accuracy:", acc_manual)

models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=200)
}

best_acc = 0
best_model_name = ""

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"{name} Accuracy: {acc:.4f}")
    
    if acc > best_acc:
        best_acc = acc
        best_model_name = name

print("\nBest Model (AutoML Simulation):", best_model_name)
print("Best Accuracy:", best_acc)

labels = ["Manual", "AutoML Best"]
values = [acc_manual, best_acc]

plt.figure()
plt.bar(labels, values)
plt.title("Manual vs AutoML (Simulated)")
plt.ylabel("Accuracy")
plt.show()