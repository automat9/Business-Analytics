import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# -------------------------------
# 1. Load and Preprocess the Dataset
# -------------------------------
df = pd.read_excel("https://github.com/automat9/Business-Analytics/raw/master/Semester%202/Machine%20Learning%20for%20Social%20Data%20Science/Assignments/car%20prices%20final.xlsx")
cat_cols = ['model', 'trim', 'body', 'transmission', 'color', 'interior']

# One-hot encode categorical features
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Define features and target variable
X = df.drop('sellingprice', axis=1)
y = df['sellingprice']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# -------------------------------
# 2. Log-Transformed Linear Regression Model
# -------------------------------
# Apply log transformation to the target variable
y_train_log = np.log(y_train)
y_test_log = np.log(y_test)

# Train the model on the log-transformed target
lr_log = LinearRegression().fit(X_train, y_train_log)

# Predict on the test set and convert predictions back to the original scale
y_pred_log = lr_log.predict(X_test)
y_pred = np.exp(y_pred_log)  # renamed for clarity

# Evaluate the model
mse_log = mean_squared_error(y_test, y_pred)
r2_log = r2_score(y_test, y_pred)
n = X_test.shape[0]  # number of observations
p = X_test.shape[1]  # number of predictors
adjusted_r2_log = 1 - (1 - r2_log) * (n - 1) / (n - p - 1)

print("Linear Regression with Log-Transformed Target:")
print("  Mean Squared Error:", mse_log)
print("  R^2:", r2_log)
print("  Adjusted R^2:", adjusted_r2_log)

# -------------------------------
# 3. Predicting Unseen Data Point
# -------------------------------
new_car_data = {
    'year': [2005],
    'model': ['Explorer'],       
    'trim': ['XLT'],           
    'body': ['SUV'],          
    'transmission': ['automatic'],
    'condition': [45],
    'odometer': [130000],
    'color': ['black'],           
    'interior': ['gray']}

new_car_df = pd.DataFrame(new_car_data)

# One-hot encode the new data point and reindex to match training data
new_car_encoded = pd.get_dummies(new_car_df, columns=cat_cols, drop_first=True)
new_car_encoded = new_car_encoded.reindex(columns=X_train.columns, fill_value=0).astype(float)

# Predict the selling price for the new car and convert back to original scale
predicted_price_log = np.exp(lr_log.predict(new_car_encoded))[0]
print("Predicted Selling Price for the Previously Unseen Car (Log-Transformed Model): ${:.2f}".format(predicted_price_log))

#################################################
# Visualisation 1: Actual vs. Predicted Values
#################################################
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs. Predicted Selling Price")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)  # perfect prediction line
plt.show()

#################################################
# Visualisation 2: Residual Plot
#################################################
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.7)
plt.xlabel("Predicted Selling Price")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.axhline(0, color='red', linestyle='--', lw=2)
plt.show()

#################################################
# Visualisation 3: Partial Dependence on 'odometer'
#################################################
# Create a range for the 'odometer' feature based on X_test values
odometer_range = np.linspace(X_test['odometer'].min(), X_test['odometer'].max(), 100)

# Create a DataFrame of average feature values from the test set
avg_features = X_test.mean().to_frame().T

# Repeat the average values to match the length of odometer_range and update 'odometer'
partial_data = pd.concat([avg_features]*100, ignore_index=True)
partial_data['odometer'] = odometer_range

# Predict using the log-transformed model and convert predictions back to the original scale
partial_preds = np.exp(lr_log.predict(partial_data))

plt.figure(figsize=(8, 6))
plt.plot(odometer_range, partial_preds)
plt.xlabel("Odometer")
plt.ylabel("Predicted Selling Price")
plt.title("Partial Dependence of Selling Price on Odometer")
plt.show()
