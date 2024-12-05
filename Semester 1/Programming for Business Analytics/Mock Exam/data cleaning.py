# data cleaning
df.isnull() # Check for missing values (returns a boolean DataFrame).
df.dropna() # Remove rows/columns with missing values.
df.fillna(value) # Fill missing values with a specified value.

import pandas as pd

# Data
data = {
    'Name': ['Alice', 'Bob', None, 'David'],
    'Age': [25, 30, 22, None],
    'Score': [85, 90, None, 88]
}
df = pd.DataFrame(data)

# Fill missing values in 'Name' with 'Unknown'
df['Name'] = df['Name'].fillna('Unknown')

# Calculate the mean of 'Age' and 'Score' columns
age_mean = df['Age'].mean()
score_mean = df['Score'].mean()

# Fill missing values in 'Age' and 'Score' columns with their respective means
df['Age'] = df['Age'].fillna(age_mean)
df['Score'] = df['Score'].fillna(score_mean)

print(df)
