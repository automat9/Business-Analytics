#######################################################################################################################################################
# Total: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
# To do: 5, 6, 9
# Name: Matt Paw
# SID: 730068016
# Exam Date: N/A
# Module: Programming for Business Analytics
# Github link for this assignment: 
# https://github.com/automat9/Business-Analytics/blob/master/Semester%201/Programming%20for%20Business%20Analytics/Mock%20Exam/2.answers.py
#######################################################################################################################################################
# Instruction 1. Read each question carefully and complete the scripts as instructed.

# Instruction 2. Only ethical and minimal use of AI is allowed. You may use AI to get advice on tool usage or language syntax, 
#                but not to generate code. Clearly indicate how and where you used AI.

# Instruction 3. Include comments explaining the logic of your code and the output as a comment below the code.

# Instruction 4. Commit to Git and upload to ELE once you finish.

#######################################################################################################################################################

# Question 1 - Loops and Lists
# You are given a list of numbers representing weekly sales in units.
weekly_sales = [120, 85, 100, 90, 110, 95, 130]

# Write a for loop that iterates through the list and prints whether each week's sales were above or below the average sales for the period.
# Calculate and print the average sales.

average = sum(weekly_sales)/7
print('The average is:', average)

for i in weekly_sales:
    if i>average:
        print(i, "is above average")
    else:
        print(i, "is below average")


#######################################################################################################################################################

# Question 2 - String Manipulation
# A customer feedback string is provided:
customer_feedback = """The product was good but could be improved. I especially appreciated the customer support and fast response times."""

# Find the first and last occurrence of the words 'good' and 'improved' in the feedback using string methods.
# Store each position in a list as a tuple (start, end) for both words and print the list.
good = ((customer_feedback.find('good'), customer_feedback.find('good')+len('good')))
improved = ((customer_feedback.find('improved'), customer_feedback.find('improved')+len('improved')))
print('The coordinates of the word good are:', good)
print('The coordinates of the word improved are:', improved)
#######################################################################################################################################################

# Question 3 - Functions for Business Metrics
# Define functions to calculate the following metrics, and call each function with sample values (use your student ID digits for customization).

# 1. Net Profit Margin: Calculate as (Net Profit / Revenue) * 100.
# 2. Customer Acquisition Cost (CAC): Calculate as (Total Marketing Cost / New Customers Acquired).
# 3. Net Promoter Score (NPS): Calculate as (Promoters - Detractors) / Total Respondents * 100.
# 4. Return on Investment (ROI): Calculate as (Net Gain from Investment / Investment Cost) * 100.

#1.
def net_profit_margin(net_profit,revenue):
    return (net_profit / revenue)*100
    
print('Net Profit Margin is:', net_profit_margin(68016,730068016))

#2.
def CAC(total_marketing_cost, new_customers):
    return (total_marketing_cost/new_customers)
print('Customer Acquisition Cost is:', CAC(730068016, 68016))

#3.
def NPS(promoters, detractors, total_respondents):
    return (promoters - detractors) / total_respondents * 100
print('Net Promoter Score is:', NPS(730, 16, 68016))

#4.
def ROI(net_gain,investment_cost):
    return (net_gain / investment_cost) * 100
print('Return on Investment is:', ROI(68016, 7300))
#######################################################################################################################################################

# Question 4 - Data Analysis with Pandas
# Using a dictionary sales_data, create a DataFrame from this dictionary, and display the DataFrame.
# Write code to calculate and print the cumulative monthly sales up to each month.
import pandas as pd

sales_data = {'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May'], 'Sales': [200, 220, 210, 240, 250]}

sales_df = pd.DataFrame(sales_data)

sales_df['Cummulative'] = sales_df['Sales'].cumsum()
print(sales_df)
#######################################################################################################################################################

# Question 5 - Linear Regression for Forecasting
# Using the dataset below, create a linear regression model to predict the demand for given prices.
# Predict the demand if the company sets the price at £26. Show a scatter plot of the data points and plot the regression line.

# Price (£): 15, 18, 20, 22, 25, 27, 30
# Demand (Units): 200, 180, 170, 160, 150, 140, 130

#######################################################################################################################################################

# Question 6 - Error Handling
# You are given a dictionary of prices for different products.
prices = {'A': 50, 'B': 75, 'C': 'unknown', 'D': 30}

# Write a function to calculate the total price of all items, handling any non-numeric values by skipping them.
# Include error handling in your function and explain where and why it’s needed.

#######################################################################################################################################################

# Question 7 - Plotting and Visualization
# Generate 50 random numbers between 1 and 500, then:
# Plot a histogram to visualize the distribution of these numbers.
# Add appropriate labels for the x-axis and y-axis, and include a title for the histogram.

import matplotlib.pyplot as plt
import random

sample = random.sample(range(1,501),50)
plt.xlabel('Numbers')
plt.ylabel('Frequency')
plt.title('Histogram showing the frequency of 50 random numbers between 1 and 500') 

plt.hist(sample)
plt.show()

#######################################################################################################################################################

# Question 8 - List Comprehensions
# Given a list of integers representing order quantities.
quantities = [5, 12, 9, 15, 7, 10]

# Use a list comprehension to create a new list that doubles each quantity that is 10 or more.
new_quantities = [i * 2 for i in quantities if i>=10]
# Print the original and the new lists.
print('Old Quantities for Q8:', quantities)
print('New Quantities for Q8:', new_quantities)
#######################################################################################################################################################

# Question 9 - Dictionary Manipulation
# Using the dictionary below, filter out the products with a rating of less than 4 and create a new dictionary with the remaining products.
ratings = {'product_A': 4, 'product_B': 5, 'product_C': 3, 'product_D': 2, 'product_E': 5}

#######################################################################################################################################################

# Question 10 - Debugging and Correcting Code
# The following code intends to calculate the average of a list of numbers, but it contains errors:
values = [10, 20, 30, 40, 50]
total = 0
for i in values:
    total = total + i

average = total / len(values)
print("Fixed Q9: The average is", average) # remove the + between the string and the float and replace with a comma

# Identify and correct the errors in the code.
# Comment on each error and explain your fixes.

#######################################################################################################################################################
