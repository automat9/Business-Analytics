# Finding better LR using existing data point

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import matplotlib.pyplot as plt
import statsmodels.api as sm

# -------------------------------
# 1. Load and Preprocess the Dataset
# -------------------------------
df = pd.read_excel("https://github.com/automat9/Business-Analytics/raw/master/Semester%202/Machine%20Learning%20for%20Social%20Data%20Science/Assignments/car%20prices%20final.xlsx")
cat_cols = ['model', 'trim', 'body', 'transmission', 'color', 'interior']  # categorical columns

# One-hot encode categorical features
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Define features and target
X = df.drop('sellingprice', axis=1)
y = df['sellingprice']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# -------------------------------
# 2. Base Linear Regression Model
# -------------------------------
lr_model = LinearRegression().fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
print("Base Linear Regression:")
print("  MSE:", mse_lr)
print("  R^2:", r2_lr)

# -------------------------------
# 3. Linear Regression with Polynomial Features (degree=2)
# -------------------------------
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

lr_poly = LinearRegression().fit(X_train_poly, y_train)
y_pred_poly = lr_poly.predict(X_test_poly)
mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)
print("\nPolynomial Linear Regression (degree=2):")
print("  MSE:", mse_poly)
print("  R^2:", r2_poly)

# -------------------------------
# 4. Linear Regression with Log-Transformed Target
# -------------------------------
# Transform the target variable using log
y_train_log = np.log(y_train)
y_test_log = np.log(y_test)

lr_log = LinearRegression().fit(X_train, y_train_log)
y_pred_log = lr_log.predict(X_test)
# Convert predictions back to original scale
y_pred_log_exp = np.exp(y_pred_log)
mse_log = mean_squared_error(y_test, y_pred_log_exp)
r2_log = r2_score(y_test, y_pred_log_exp)
print("\nLinear Regression with Log-Transformed Target:")
print("  MSE:", mse_log)
print("  R^2:", r2_log)

# -------------------------------
# 5. Regularized Models: Ridge and Lasso
# -------------------------------
ridge = Ridge(alpha=1.0).fit(X_train, y_train)
lasso = Lasso(alpha=0.1, max_iter=10000).fit(X_train, y_train)

y_pred_ridge = ridge.predict(X_test)
y_pred_lasso = lasso.predict(X_test)

mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

print("\nRidge Regression:")
print("  MSE:", mse_ridge)
print("  R^2:", r2_ridge)

print("\nLasso Regression:")
print("  MSE:", mse_lasso)
print("  R^2:", r2_lasso)

# -------------------------------
# 6. Linear Regression with Feature Scaling
# -------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_scaled = LinearRegression().fit(X_train_scaled, y_train)
y_pred_scaled = lr_scaled.predict(X_test_scaled)
mse_scaled = mean_squared_error(y_test, y_pred_scaled)
r2_scaled = r2_score(y_test, y_pred_scaled)
print("\nLinear Regression with Scaling:")
print("  MSE:", mse_scaled)
print("  R^2:", r2_scaled)

# -------------------------------
# 7. Predicting a New Data Point with Each Model
# -------------------------------
new_car_data = {
    'year': [2002],
    'model': ['Ranger'],       
    'trim': ['XLT'],           
    'body': ['regular cab'],          
    'transmission': ['automatic'],
    'condition': [28],
    'odometer': [157687],
    'color': ['white'],           
    'interior': ['brown']
}
new_car_df = pd.DataFrame(new_car_data)

# One-hot encode the new data point and reindex to match training data
new_car_encoded = (pd.get_dummies(new_car_df, columns=cat_cols, drop_first=True)
                     .reindex(columns=X_train.columns, fill_value=0)
                     .astype(float))

# Base Linear Regression Prediction
pred_base = lr_model.predict(new_car_encoded)[0]

# Polynomial Regression Prediction
new_car_poly = poly.transform(new_car_encoded)
pred_poly = lr_poly.predict(new_car_poly)[0]

# Log-Transformed Linear Regression Prediction
pred_log = np.exp(lr_log.predict(new_car_encoded))[0]

# Ridge Regression Prediction
pred_ridge = ridge.predict(new_car_encoded)[0]

# Lasso Regression Prediction
pred_lasso = lasso.predict(new_car_encoded)[0]

# Scaled Linear Regression Prediction
new_car_scaled = scaler.transform(new_car_encoded)
pred_scaled = lr_scaled.predict(new_car_scaled)[0]

print("\nPredicted Selling Prices for the Existing Data Point:")
print("  Base Linear Regression:       ${:.2f}".format(pred_base))
print("  Polynomial Regression:        ${:.2f}".format(pred_poly))
print("  Log-Transformed Regression:   ${:.2f}".format(pred_log))
print("  Ridge Regression:             ${:.2f}".format(pred_ridge))
print("  Lasso Regression:             ${:.2f}".format(pred_lasso))
print("  Scaled Linear Regression:     ${:.2f}".format(pred_scaled))
