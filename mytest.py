import pandas as pd
from tabulate import tabulate
import numpy as np

### Part 1   ###
###  what is for label, m in df.groupby('label') ###
# Create a sample DataFrame
# data = {
#     'label': ['A', 'B', 'A', 'B', 'A', 'B'],
#     'value': [10, 20, 30, 40, 50, 60]
# }
# df = pd.DataFrame(data)

# # Group by the 'label' column
# for label, m in df.groupby('label'):
#     print(f"Label: {label}")
#     print(tabulate(m, headers='keys', tablefmt='psql'))
#     print()


###################################
### Part 2 ###
### what is y[m.index, None]

# Example y array
y = np.array([10, 20, 30, 40, 50])

# Example meta DataFrame
meta = pd.DataFrame({
    'label': ['A', 'B', 'A', 'B', 'A']
})

# Group by 'label'
grouped_meta = meta.groupby('label')

# Select group 'A'
m = grouped_meta.get_group('A')

# Indices for group 'A'
print(m.index)  # Output: Int64Index([0, 2, 4], dtype='int64')
print("#############")

print(y[m.index])
print("#############")
# Select and reshape y
selected_y = y[m.index, None]
print(selected_y)
# Output:
# array([[10],
#        [30],
#        [50]])