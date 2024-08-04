import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

### demo for kfold in binary classification

# Step 1: Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# For simplicity, we'll convert this to a binary classification problem
# by selecting only two classes

# X is expected to have 4 features

X = X[y != 2]
y = y[y != 2]
print("shape of X")
print(X.shape)
print("X:")
print(X)

print("--------")
print("shape of y")
print(y.shape)
print("y:")
print(y)

# Step 2: Define the model
model = make_pipeline(StandardScaler(), LogisticRegression())

# Step 3: Set up KFold
cv = KFold(n_splits=5, shuffle=True, random_state=0)

# Step 4: Use cross_val_predict
preds = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]

# Print the predictions
# print(preds)

# Create a DataFrame to display the ground truth and predictions
results_df = pd.DataFrame({
    'Ground Truth': y,
    'Predicted Probability of Class 1': preds
})

# Print the DataFrame
print(tabulate(results_df, headers='keys', tablefmt='github'))
exit(0)
# Optional: Visualize the predictions
plt.scatter(range(len(preds)), preds, c=y, cmap='bwr', alpha=0.7)
plt.xlabel('Sample index')
plt.ylabel('Predicted probability of class 1')
plt.title('Cross-validated predictions')
plt.show()