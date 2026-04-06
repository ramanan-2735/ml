from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

digits = load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train, y_train)
y_pred_linear = svm_linear.predict(X_test)
acc_linear = accuracy_score(y_test, y_pred_linear)

svm_poly = SVC(kernel='poly', degree=3)
svm_poly.fit(X_train, y_train)
y_pred_poly = svm_poly.predict(X_test)
acc_poly = accuracy_score(y_test, y_pred_poly)

svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(X_train, y_train)
y_pred_rbf = svm_rbf.predict(X_test)
acc_rbf = accuracy_score(y_test, y_pred_rbf)

print("SVM Kernel Comparison Results:\n")
print(f"Linear Kernel Accuracy: {acc_linear:.4f}")
print(f"Polynomial Kernel Accuracy: {acc_poly:.4f}")
print(f"RBF Kernel Accuracy: {acc_rbf:.4f}")

kernels = ['Linear', 'Polynomial', 'RBF']
accuracies = [acc_linear, acc_poly, acc_rbf]

plt.figure()
plt.bar(kernels, accuracies)
plt.xlabel("Kernel Type")
plt.ylabel("Accuracy")
plt.title("SVM Kernel Performance Comparison on Digits Dataset")
plt.show()