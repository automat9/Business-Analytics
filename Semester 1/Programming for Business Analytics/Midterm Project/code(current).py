# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt

# Load data from GitHub URL and reorder columns
url = "https://github.com/automat9/Business-Analytics/raw/aadac54eaf6f4ca76c4970a1d317ad355a7fe051/Semester%201/Programming%20for%20Business%20Analytics/Midterm%20Project/Coffee_company.csv"
data = pd.read_csv(url)

# Remove white space
data.columns = data.columns.str.strip()

# Rename column
data.rename(columns={'Month Name':'Month'}, inplace = True)

# Floats into whole numbers
data['Units Sold'] = data['Units Sold'].astype(int)

# Total sales
data['Sales'] = pd.to_numeric(data['Sales'].str.replace(',','', regex=False).str.replace('$','', regex=False), errors='coerce')
data['Sales'] = data['Sales'].astype(float)
total_sales = data['Sales'].sum()
print('Total Sales:' '$', total_sales)

# Quarterly sales
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
data.set_index('Date', inplace=True)
# Calculate
quarterly_sales = data['Sales'].resample('Q').sum().reset_index()
quarterly_sales.columns = ['Quarter', 'Total Sales']
print(quarterly_sales)

# Plot a bar graph
plt.figure(figsize=(10, 6))
plt.bar(quarterly_sales['Quarter'].dt.to_period('Q').astype(str), quarterly_sales['Total Sales'], color='skyblue')
plt.title('Quarterly Sales')
plt.xlabel('Quarter')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


#data.head()
