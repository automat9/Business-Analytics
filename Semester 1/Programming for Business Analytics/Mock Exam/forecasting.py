# Forecasting 1
import statsmodels.api as sma
import matplotlib.pyplot as plt
budget = [10,15,29,25,30] # independent
sales = [25,30,40,50,60] # dependent

model = sma.OLS(sales, sma.add_constant(budget)) # ordinary least squares, first what we're trying to predict
result = model.fit()
#print(result.summary())

plt.scatter(budget, sales, label='Actual values') # budget on x axis because independent, dependend on y
plt.plot(budget, result.predict(), color='r', label='Predicted') # predictions for given budget values
plt.xlabel('budget')
plt.ylabel('sales')
plt.legend()
plt.show()

predicted_sales = result.predict([1,35])
print('The predicted sales at $35 is:', predicted_sales[0])




# Forecasting 2
day = [1,2,3,4,5] # independent
temperature = [15,17,19,21,23] # dependent

model = sma.OLS(temperature, sma.add_constant(day))
result = model.fit()
#print(result.summary())

plt.scatter(day, temperature, label='actual values')
plt.plot(day, result.predict(),color='r', label='predicted')
plt.xlabel('day')
plt.ylabel('temperature')
plt.legend()
plt.show()

temperature_day6 = result.predict([1,6])
temperature_day7 = result.predict([1,7])
print('Predicted temperature for day 6 is:', temperature_day6)
print('Predicted temperature for day 7 is:', temperature_day7)
