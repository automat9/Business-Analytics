# 1
import matplotlib.pyplot as plt

products = ['A', 'B', 'C', 'D', 'E']
sales = [50, 70, 30, 85, 60]

plt.bar(products, sales)
plt.title('Product Sales Comparison')
plt.xlabel('Products')
plt.ylabel('Sales')
plt.show()

# 2
day = [1,2,3,4,5]
visitors = [100,150,200,250,300]

plt.plot(day, visitors)
plt.ylabel('Day')
plt.xlabel('Visitors')
plt.title('Number of visitors per day')
plt.grid()
plt.show()

# Line Plot: plt.plot()
# Bar Chart: plt.bar()
# Horizontal Bar Chart: plt.barh()
# Scatter Plot: plt.scatter()
# Histogram: plt.hist()
# Pie Chart: plt.pie()
# Box Plot: plt.boxplot()

