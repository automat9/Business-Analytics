# Log-transformed RF
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

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

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# -------------------------------
# 2. Random Forest with Log-Transformed Target
# -------------------------------
# Apply log transformation to the target variable
y_train_log = np.log(y_train)
y_test_log = np.log(y_test)

# Train the model on the log-transformed target
rf_log = RandomForestRegressor(n_estimators=100, random_state=42)
rf_log.fit(X_train, y_train_log)

# Predict on the test set and convert predictions back to the original scale
y_pred_log = rf_log.predict(X_test)
y_pred_log_exp = np.exp(y_pred_log)

# Evaluate the model
mse_log = mean_squared_error(y_test, y_pred_log_exp)
r2_log = r2_score(y_test, y_pred_log_exp)
n = X_test.shape[0]  # Number of observations
p = X_test.shape[1]  # Number of predictors
adjusted_r2_log = 1 - (1 - r2_log) * (n - 1) / (n - p - 1)

print("Random Forest with Log-Transformed Target:")
print("  Mean Squared Error:", mse_log)
print("  R^2 Score:", r2_log)
print("  Adjusted R^2 Score:", adjusted_r2_log)

# -------------------------------
# 3. Predicting another Existing Data Point
# -------------------------------
new_car_data = {
    'year': [2003],
    'model': ['Mustang'],
    'trim': ['Premium'],
    'body': ['Convertible'],
    'transmission': ['automatic'],
    'condition': [33],
    'odometer': [75349],
    'color': ['black'],
    'interior': ['gray']
}
new_car_df = pd.DataFrame(new_car_data)

# One-hot encode the new data point and reindex to match training data
new_car_encoded = pd.get_dummies(new_car_df, columns=cat_cols, drop_first=True)
new_car_encoded = new_car_encoded.reindex(columns=X_train.columns, fill_value=0).astype(float)

# Predict the selling price for the new car and convert back to original scale
predicted_price_log = np.exp(rf_log.predict(new_car_encoded))[0]
print("Predicted Selling Price for another existing car (RF with Log Target): ${:.2f}".format(predicted_price_log))
