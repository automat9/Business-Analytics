# Import pandas and matplotlib
import pandas as pd
import matplotlib.pyplot as plt

# Load the metadata
data = pd.read_csv("https://raw.githubusercontent.com/automat9/Business-Analytics/refs/heads/master/Semester%201/Topics%20in%20Business%20Analytics/runs/results/2results.csv")

# Remove white space from column names
data.columns = data.columns.str.strip()

# Plot config
plt.figure(figsize=(10, 5))
plt.plot(data["metrics/mAP_0.5:0.95"], label="mAP_0.5:0.95")
plt.xlabel("epoch")
plt.ylabel("mAP")
plt.grid(True, alpha=1)
plt.title("mAP_0.5:0.95 for Second Run")
plt.legend()

# Show the plot
plt.show()

