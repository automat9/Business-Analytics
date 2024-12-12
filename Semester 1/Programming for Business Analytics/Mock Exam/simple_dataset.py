import pandas as pd
import matplotlib.pyplot as plt
import stats models.api

df = pd.read_csv('simple_dataset.csv')

df.isnull().sum()

df = df.dropna()

def net_profit(sales, profit, discount):
    return sales - (profit + discount)

sales = sum(df['Sales'])
profit = sum(df['Profit'])
discount = sum(df['Discount'])

net_profit(sales, profit, discount)

