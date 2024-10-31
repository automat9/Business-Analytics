import pandas as pd
import matplotlib as plt
import numpy as np

data = pd.read_csv(r"C:\Users\10\Desktop\Coffee_company.csv")

#print(data.shape) # rows & columns

#print(data.columns) # names of columns

# Reordering columns
new_order = [
    'Date', 'Year', 'Month Number', ' Month Name ',
    'Segment', 'Country', ' Product ', ' Discount Band ',
    'Units Sold', ' Manufacturing Price ', ' Sale Price ',
    ' Gross Sales ', ' Discounts ', '  Sales ', ' COGS ', ' Profit '
]
# Assigning the reordered columns to a new variable sorted_data
sorted_data = data[new_order]

sorted_data.head()

sorted_data
