# Import relevant libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load and preprocess the dataset
df = pd.read_excel("https://github.com/automat9/Business-Analytics/raw/master/Semester%202/Machine%20Learning%20for%20Social%20Data%20Science/Assignments/car%20prices%20final.xlsx")
cat_cols = ['model', 'trim', 'body', 'transmission', 'color', 'interior'] # listing non numerical columns
df = pd.get_dummies(df, columns=cat_cols, drop_first=True) # encoding non numerical data (critical for regression analysis)

# Define features and target variable
X = df.drop('sellingprice', axis=1) # dependent (feature) variables (every column except sellingprice)
y = df['sellingprice'] # independent (target) variable

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the model
lr_model = LinearRegression().fit(X_train, y_train)

# Predict on the test set
y_pred = lr_model.predict(X_test)

# Calculate and print metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)  # Save R² score into a variable
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

# Compute Adjusted R^2 for the test set
n = X_test.shape[0]  # Number of observations in test set
p = X_test.shape[1]  # Number of predictors
adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
print("Adjusted R^2 Score:", adjusted_r2)

# Define and preprocess new car data for prediction
new_car_data = {
    'year': [2000],
    'model': ['Explorer'],       
    'trim': ['XLS'],           
    'body': ['SUV'],          
    'transmission': ['automatic'],
    'condition': [40],
    'odometer': [120000],
    'color': ['red'],           
    'interior': ['black']}

# Create a DataFrame from the new car data dictionary
new_car_df = pd.DataFrame(new_car_data)

# One-hot encode the categorical variables in the new car DataFrame,
# then reindex the resulting DataFrame to match the columns used in training (filling missing columns with 0),
# and finally convert all values to float type
new_car_encoded = (pd.get_dummies(new_car_df, columns=cat_cols, drop_first=True)
                     .reindex(columns=X_train.columns, fill_value=0)
                     .astype(float))

# Use the trained linear regression model to predict the selling price for the new car
predicted_price = lr_model.predict(new_car_encoded)[0]
print("Predicted Selling Price:", predicted_price)


################################################# Visualisation 1
# Create a scatter plot of actual vs. predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs. Predicted Selling Price")

# Plot a red dashed line for perfect predictions
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.show()

################################################# Visualisation 2
# Calculate residuals (the differences between actual and predicted values)
residuals = y_test - y_pred

# Create a residual plot
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.7)
plt.xlabel("Predicted Selling Price")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.axhline(0, color='red', linestyle='--', lw=2)
plt.show()

################################################# Visualisation 3
# Choose a feature to examine, in this case 'odometer'
odometer_range = np.linspace(X_test['odometer'].min(), X_test['odometer'].max(), 100)
# Create a DataFrame based on the average of each feature in the test set
avg_features = X_test.mean().to_frame().T

# Repeat the average values for each value of the chosen feature
partial_data = pd.concat([avg_features]*100, ignore_index=True)
partial_data['odometer'] = odometer_range

# Predict using the model
partial_preds = lr_model.predict(partial_data)

plt.figure(figsize=(8, 6))
plt.plot(odometer_range, partial_preds)
plt.xlabel("Odometer")
plt.ylabel("Predicted Selling Price")
plt.title("Partial Dependence of Selling Price on Odometer")
plt.show()
