from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=100, n_features=2, n_informative=2,
                           n_redundant=0, n_classes=2, weights=[0.2, 0.8],
                           class_sep=0.95, random_state=0)
plt.figure(figsize=(12, 8))
plt.title('Repartition before SMOTE')
plt.scatter(X[y==1][:, 0], X[y==1][:, 1], label='class 1')
plt.scatter(X[y==0][:, 0], X[y==0][:, 1], label='class 0')
plt.legend()
plt.grid(False)
plt.show()

smt = SMOTE()
X_smote, y_smote = smt.fit_resample(X, y)

plt.figure(figsize=(12, 8))
plt.title('Repartition after SMOTE')
plt.scatter(X_smote[y_smote==1][:, 0], X_smote[y_smote==1][:, 1], label='class 1')
plt.scatter(X_smote[y_smote==0][:, 0], X_smote[y_smote==0][:, 1], label='class 0')
plt.legend()
plt.grid(False)
plt.show()