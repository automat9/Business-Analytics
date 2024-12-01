"""
This is how
you can add comments
that are not just one liners
"""

# Loop
weekly_sales = [120, 85, 100, 90, 110, 95, 130]
average = sum(weekly_sales)/7
print('The average is:', average)
for i in weekly_sales:
    if i>average:
        print(i, 'is above average')
    else:
        print(i, 'is below average')
#######################################################################################################################################################
# String Manipulation
string = """The product was good but could be improved. I especially appreciated the customer support and fast response times."""
good = ((string.find('good'), string.find('good')+len('good')))
improved = ((string.find('improved'), string.find('improved')+len('improved')))
print('The coordinates of the word good are:', good)
print('The coordinates of the word improved are:', improved)
#######################################################################################################################################################
# Functions
def x(a, b, c):
    return ((a - b) / c) * 100
print('The value of x:', x(4, 5, 2))
#######################################################################################################################################################
# Data Analysis with Pandas
import pandas as pd
sales_data = {'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May'], 'Sales': [200, 220, 210, 240, 250]}
sales_df = pd.DataFrame(sales_data)
sales_df['Cummulative'] = sales_df['Sales'].cumsum()
print(sales_df)
#######################################################################################################################################################
# Forecasting - linear regression model
price = [15, 18, 20, 22, 25, 27, 30]
demand = [200, 180, 170, 160, 150, 140, 130]
import statsmodels.api as sma
import matplotlib.pyplot as plt
# Regression Model
model = sma.OLS(demand, sma.add_constant(price))
result = model.fit()
print(result.summary())
# Scatter plot
plt.scatter(price, demand, label='Actual Data')
plt.plot(price, result.predict(), color='r', label='Predicted')
plt.xlabel('Price')
plt.ylabel('Demand')
plt.title('Scatter diagram showing the relation between Price and Demand')
plt.legend()
plt.show()
# Predict demand at 26 pounds
predicted_demand = result.predict([1,26])
print('The predicted demand at price £26 is:', predicted_demand[0])
#######################################################################################################################################################
# Error Handling
prices = {'A': 50, 'B': 75, 'C': 'unknown', 'D': 30}
def total_price(prices):
    total = 0 
    for product, price in prices.items():
        try:
            total += price
        except ValueError:
            print('Skipping non-numeric value for item:', product, price)
        except TypeError:
            print('Skipping invalid type for item:', product, price)
    return total
print(total_price(prices))
#######################################################################################################################################################
# Question 7 - Plotting and Visualization
import matplotlib.pyplot as plt
import random
sample = random.sample(range(1,501),50)
plt.hist(sample)
plt.xlabel('Numbers')
plt.ylabel('Frequency')
plt.title('Histogram showing the frequency of 50 random numbers between 1 and 500') 
plt.show()
#######################################################################################################################################################
# List Comprehensions
quantities = [5, 12, 9, 15, 7, 10]
# doubles each quantity that is 10 or more.
new_quantities = [i * 2 for i in quantities if i>=10]
print('Old Quantities for Q8:', quantities)
print('New Quantities for Q8:', new_quantities)
#######################################################################################################################################################
# Dictionary Manipulation
# Delete ratings of less than 4
ratings = {'product_A': 4, 'product_B': 5, 'product_C': 3, 'product_D': 2, 'product_E': 5}
filtered_ratings = {product: rating for product, rating in ratings.items() if rating >= 4}
print('The filtered ratings from Q9:', filtered_ratings)
#######################################################################################################################################################
# Debugging and Correcting Code
values = [10, 20, 30, 40, 50]
total = 0
for i in values:
    total = total + i
average = total / len(values)
print('Fixed Q9: The average is', average)
########################################################################################################################################################
