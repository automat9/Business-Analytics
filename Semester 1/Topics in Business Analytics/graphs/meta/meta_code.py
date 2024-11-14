# Import pandas and matplotlib
import pandas as pd
import matplotlib.pyplot as plt

# Load the metadata and remove white space from column names
data = pd.read_csv("https://raw.githubusercontent.com/automat9/Business-Analytics/refs/heads/master/Semester%201/Topics%20in%20Business%20Analytics/runs/results/results_metadata.csv")
data.columns = data.columns.str.strip()

###### Graph Setup ######
plt.figure(figsize=(10, 6))
colours = {"first": "blue","second": "orange","third": "green"}
    
# Plot the grapgh
for column in data.columns[1:]: # Skip firt "epoch" column
    group = column.strip().split('/')[0]  # Extract groups
    plt.plot(data["epoch"], data[column], label=column, color=colours[group], linewidth=2)

    
# Plot config
plt.xlabel("Epoch")
plt.ylabel("mAP Score")
plt.title("mAP Scores over Epochs")
plt.grid(alpha=1)
plt.legend()

# Show the plot
plt.show()
