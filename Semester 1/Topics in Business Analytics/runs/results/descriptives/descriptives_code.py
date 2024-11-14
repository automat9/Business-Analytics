# Import pandas
import pandas as pd

# Load the metadata
data = pd.read_csv("https://raw.githubusercontent.com/automat9/Business-Analytics/refs/heads/master/Semester%201/Topics%20in%20Business%20Analytics/runs/results/results_metadata.csv")

# Remove white space from column names
data.columns = data.columns.str.strip()

# Sort columns for each run
first = data[["first/mAP_0.5", "first/mAP_0.5:0.95"]].describe()
second = data[["second/mAP_0.5", "second/mAP_0.5:0.95"]].describe()
third = data[["third/mAP_0.5", "third/mAP_0.5:0.95"]].describe()

# Display the statistics
print("Descriptive Statistics for First Run:")
print(first)

print("Descriptive Statistics for Second Run:")
print(second)

print("Descriptive Statistics for Third Run:")
print(third)
