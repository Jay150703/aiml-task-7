# Task 7: Support Vector Machines (SVM)
# Objective: Use SVMs for linear and non-linear classification.

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset (Breast Cancer)
data = datasets.load_breast_cancer()
X, y = data.data, data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Linear SVM
linear_svm = SVC(kernel='linear', C=1, random_state=42)
linear_svm.fit(X_train, y_train)
y_pred_linear = linear_svm.predict(X_test)

print("Linear SVM Classification Report:")
print(classification_report(y_test, y_pred_linear))

# Confusion Matrix for Linear SVM
print("Confusion Matrix (Linear SVM):")
print(confusion_matrix(y_test, y_pred_linear))

# RBF Kernel SVM
rbf_svm = SVC(kernel='rbf', C=1, gamma=0.01, random_state=42)
rbf_svm.fit(X_train, y_train)
y_pred_rbf = rbf_svm.predict(X_test)

print("\nRBF SVM Classification Report:")
print(classification_report(y_test, y_pred_rbf))

# Confusion Matrix for RBF SVM
print("Confusion Matrix (RBF SVM):")
print(confusion_matrix(y_test, y_pred_rbf))

# Cross-validation accuracy
scores = cross_val_score(rbf_svm, X, y, cv=5)
print("\nCross-validation accuracy (RBF SVM):", scores.mean())

# --- Visualization on a 2D dataset (for decision boundaries) ---
X_vis, y_vis = datasets.make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=42, n_clusters_per_class=1
)
X_vis = StandardScaler().fit_transform(X_vis)

# Fit linear and RBF SVM on 2D data
svm_linear_vis = SVC(kernel='linear', C=1).fit(X_vis, y_vis)
svm_rbf_vis = SVC(kernel='rbf', C=1, gamma=0.5).fit(X_vis, y_vis)

# Plot decision boundaries
def plot_decision_boundary(model, X, y, title):
    h = .02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.title(title)
    plt.show()

plot_decision_boundary(svm_linear_vis, X_vis, y_vis, "Linear SVM Decision Boundary")
plot_decision_boundary(svm_rbf_vis, X_vis, y_vis, "RBF SVM Decision Boundary")
