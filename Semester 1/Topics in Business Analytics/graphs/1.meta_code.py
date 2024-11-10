# Import pandas and matplotlib
import pandas as pd
import matplotlib.pyplot as plt

# Load the metadata
data = pd.read_csv("https://raw.githubusercontent.com/automat9/Business-Analytics/214361ecbd0b86e0b7e38cf8ef66fff217dd79a3/Semester%201/Topics%20in%20Business%20Analytics/results/results_metadata.csv")

# Remove white space from column names
data.columns = data.columns.str.strip()

###### Graph ######
plt.figure(figsize=(10, 6))

# Define colors for each group
colours = {"first": "blue","second": "orange","third": "green"}
    
for column in data.columns[1:]: # Skip firt "epoch" column
    group = column.strip().split('/')[0]  # Extract groups
    plt.plot(data["epoch"], data[column], label=column, color=colors[group], linewidth=2)

    
# Plot config
plt.xlabel("Epoch")
plt.ylabel("mAP Score")
plt.title("mAP Scores over Epochs")
plt.grid(True, linestyle="-", alpha=0.9)
plt.legend()

# Show the plot
plt.show()
