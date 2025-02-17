# Finding better RF using existing data point
# Import relevant libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# -------------------------------
# 1. Load and Preprocess the Dataset
# -------------------------------
df = pd.read_excel("https://github.com/automat9/Business-Analytics/raw/master/Semester%202/Machine%20Learning%20for%20Social%20Data%20Science/Assignments/car%20prices%20final.xlsx")

# Define categorical columns that require encoding
cat_cols = ['model', 'trim', 'body', 'transmission', 'color', 'interior']

# Encode non-numerical data using get_dummies and drop the first level for each category
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Define features and target variable
X = df.drop('sellingprice', axis=1)  # All columns except the target
y = df['sellingprice']               # Target variable

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# -------------------------------
# 2. Base Random Forest Regressor
# -------------------------------
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
n = X_test.shape[0]  # Number of observations
p = X_test.shape[1]  # Number of predictors
adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print("Base Random Forest Model:")
print("  Mean Squared Error:", mse)
print("  R^2 Score:", r2)
print("  Adjusted R^2 Score:", adjusted_r2)

# Define new car data for prediction
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

# Encode the new data using the same encoding scheme and column order as the training set
new_car_encoded = pd.get_dummies(new_car_df, columns=cat_cols, drop_first=True)
new_car_encoded = new_car_encoded.reindex(columns=X_train.columns, fill_value=0).astype(float)

predicted_price_base = rf_model.predict(new_car_encoded)[0]
print("Predicted Selling Price for existing car (Base RF): ${:.2f}".format(predicted_price_base))


# -------------------------------
# 3. Random Forest with Log-Transformed Target
# -------------------------------
# Apply log transformation to the target variable
y_train_log = np.log(y_train)
y_test_log = np.log(y_test)

rf_log = RandomForestRegressor(n_estimators=100, random_state=42)
rf_log.fit(X_train, y_train_log)

# Predict on the test set and convert predictions back to the original scale
y_pred_log = rf_log.predict(X_test)
y_pred_log_exp = np.exp(y_pred_log)
mse_log = mean_squared_error(y_test, y_pred_log_exp)
r2_log = r2_score(y_test, y_pred_log_exp)
adjusted_r2_log = 1 - (1 - r2_log) * (n - 1) / (n - p - 1)

print("\nRandom Forest with Log-Transformed Target:")
print("  Mean Squared Error:", mse_log)
print("  R^2 Score:", r2_log)
print("  Adjusted R^2 Score:", adjusted_r2_log)

predicted_price_log = np.exp(rf_log.predict(new_car_encoded))[0]
print("Predicted Selling Price for existing car (RF with Log Target): ${:.2f}".format(predicted_price_log))


# -------------------------------
# 4. Hyperparameter Tuning with GridSearchCV
# -------------------------------
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf_cv = GridSearchCV(RandomForestRegressor(random_state=42),
                     param_grid, cv=5, scoring='r2', n_jobs=-1)
rf_cv.fit(X_train, y_train)

print("\nBest Parameters from GridSearchCV:", rf_cv.best_params_)
best_rf = rf_cv.best_estimator_

# Predict and evaluate tuned model
y_pred_cv = best_rf.predict(X_test)
mse_cv = mean_squared_error(y_test, y_pred_cv)
r2_cv = r2_score(y_test, y_pred_cv)
adjusted_r2_cv = 1 - (1 - r2_cv) * (n - 1) / (n - p - 1)

print("Tuned Random Forest Model:")
print("  Mean Squared Error:", mse_cv)
print("  R^2 Score:", r2_cv)
print("  Adjusted R^2 Score:", adjusted_r2_cv)

predicted_price_cv = best_rf.predict(new_car_encoded)[0]
print("Predicted Selling Price for existing car (Tuned RF): ${:.2f}".format(predicted_price_cv))
