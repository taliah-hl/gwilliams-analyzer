import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate


### demo for kfold in multi-class classification

# Step 1: Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Step 2: Define the model
model = make_pipeline(StandardScaler(), LogisticRegression(multi_class='multinomial', solver='lbfgs'))

# Step 3: Set up KFold
cv = KFold(n_splits=5, shuffle=True, random_state=0)

# Step 4: Use cross_val_predict
preds = cross_val_predict(model, X, y, cv=cv, method="predict_proba")

# Print the predictions
print(preds)

# Create a DataFrame to display the ground truth and predictions
results_df = pd.DataFrame(preds, columns=[f'Predicted Probability of Class {i}' for i in range(3)])
results_df['Ground Truth'] = y

# Print the DataFrame
print(tabulate(results_df, headers='keys', tablefmt='github'))

# Optional: Visualize the predictions
for i in range(3):
    plt.scatter(range(len(preds)), preds[:, i], label=f'Class {i}', alpha=0.7)

plt.xlabel('Sample index')
plt.ylabel('Predicted probability')
plt.title('Cross-validated predictions')
plt.legend()
plt.show()