# mean 
import pandas as pd
data = {
    'Name': ['Alice','Bob','Carol'],
    'Age': [25,22,23],
    'Score': [85,90,88]}
df = pd.DataFrame(data)
df['Score'].mean()

# filter
data = {
    'Product': ['Apple','Banana','Cherry'],
    'Price': [1.5,0.5,2.0],
    'Quantity': [10,20,15]}
df = pd.DataFrame(data)
filtered = df[df['Price']>1] 
print(filtered) # only those where price is > 1
