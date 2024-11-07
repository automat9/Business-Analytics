import pandas as pd

# Load data from GitHub URL and reorder columns
url = "https://github.com/automat9/Business-Analytics/raw/aadac54eaf6f4ca76c4970a1d317ad355a7fe051/Semester%201/Programming%20for%20Business%20Analytics/Midterm%20Project/Coffee_company.csv"
data = pd.read_csv(url)

data
