import pandas as pd
df = pd.read_csv("C:\\Users\\mp967\\Desktop\\Coffee_company.csv")
# or
df = pd.read_csv(r"C:\Users\mp967\Desktop\Coffee_company.csv")

df = pd.read_csv(r"C:\Users\mati9\Desktop\data.csv")
df = pd.read_excel(r"C:\Users\mati9\Desktop\data.xlsx")

# remove white space from columns
df.columns = df.columns.str.strip()

# check all columns for missing
missing_per_column = df.isnull().sum()

# remove rows with missing
df = df.dropna(axis=0)
# remove columns with missing
df = df.dropna(axis=1)

# removing white space, $, and ,
df['Gross Sales'] = (
    df['Gross Sales']
    .str.strip()
    .str.replace('$', '', regex=False)
    .str.replace(',','')
    .astype(float))

avg = sum(df['Gross Sales']) / len(df['Gross Sales'])
