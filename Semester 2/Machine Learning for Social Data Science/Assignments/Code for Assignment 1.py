################################################## Linear Regression ##################################################
# Import relevant libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

# Load and preprocess the dataset
df = pd.read_excel("https://github.com/automat9/Business-Analytics/raw/master/Semester%202/Machine%20Learning%20for%20Social%20Data%20Science/Assignments/car%20prices%20final.xlsx")
cat_cols = ['model', 'trim', 'body', 'transmission', 'color', 'interior'] # non numerical columns
df = pd.get_dummies(df, columns=cat_cols, drop_first=True) # encoding non-numerical data 

# Define features and target
X = df.drop('sellingprice', axis=1) # dependent (feature) variables (every column except sellingprice)
y = df['sellingprice'] # independent (target) variable

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the scikit-learn Linear Regression model
lr_model = LinearRegression().fit(X_train, y_train)

# Calculate and print metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)  # Save R² score into a variable
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

# Compute Adjusted R² for the test set
n = X_test.shape[0]  # Number of observations in test set
p = X_test.shape[1]  # Number of predictors
adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
print("Adjusted R^2 Score:", adjusted_r2)

# Define and preprocess new car data for prediction
new_car_data = {
    'year': [2000],
    'model': ['Contour'],       
    'trim': ['Base'],           
    'body': ['Sedan'],          
    'transmission': ['Automatic'],
    'condition': [20],
    'odometer': [45000],
    'color': ['red'],           
    'interior': ['black']}

new_car_df = pd.DataFrame(new_car_data)
new_car_encoded = (pd.get_dummies(new_car_df, columns=cat_cols, drop_first=True)
                     .reindex(columns=X_train.columns, fill_value=0)
                     .astype(float))

predicted_price = lr_model.predict(new_car_encoded)[0]
print("Predicted Selling Price:", predicted_price)


################################################# Visualisation 1
y_pred = lr_model.predict(X_test)

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
# Choose a feature to examine
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


################################################## Random Forest ##################################################
# Import relevant libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Load the dataset
df = pd.read_excel("https://github.com/automat9/Business-Analytics/raw/master/Semester%202/Machine%20Learning%20for%20Social%20Data%20Science/Assignments/car%20prices%20final.xlsx")

# Define categorical columns that require encoding
cat_cols = ['model', 'trim', 'body', 'transmission', 'color', 'interior']

# Encode non-numerical data using get_dummies and drop the first level for each category
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Define features and target variable
X = df.drop('sellingprice', axis=1)  # All columns except the target
y = df['sellingprice']              # Target variable

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialise and train the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test set and evaluate the model
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)  # Save R² score into a variable
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

# Compute Adjusted R²
n = X_test.shape[0]  # Number of observations
p = X_test.shape[1]  # Number of predictors
adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
print("Adjusted R^2 Score:", adjusted_r2)

# Define new car data for prediction
new_car_data = {
    'year': [2000],
    'model': ['Contour'],       
    'trim': ['Base'],           
    'body': ['Sedan'],          
    'transmission': ['Automatic'],
    'condition': [20],
    'odometer': [45000],
    'color': ['red'],           
    'interior': ['black']}

new_car_df = pd.DataFrame(new_car_data)

# Encode the new data using the same encoding scheme and column order as the training set
new_car_encoded = pd.get_dummies(new_car_df, columns=cat_cols, drop_first=True)
new_car_encoded = new_car_encoded.reindex(columns=X_train.columns, fill_value=0).astype(float)

# Predict the selling price for the new car
predicted_price = rf_model.predict(new_car_encoded)[0]
print("Predicted Selling Price for new car:", predicted_price)

################################################# Visualisation 1
# Visualise one of the trees from the Random Forest
plt.figure(figsize=(20, 10))
# Pick the first tree in the forest for visualisation
tree = rf_model.estimators_[0]
plot_tree(tree,
          feature_names=X_train.columns,
          filled=True,
          rounded=True,
          max_depth=3,   # Limiting the depth for clarity
          fontsize=10)
plt.title("Visualization of one tree from the Random Forest")
plt.show()

################################################# Visualisation 2
# Zip the features with their importance values and sort them in descending order
sorted_features_importances = sorted(zip(X_train.columns, rf_model.feature_importances_), 
                                     key=lambda x: x[1], reverse=True)

# Extract the top 20 features and their importances
top_features, top_importances = zip(*sorted_features_importances[:20])

# Create a horizontal bar plot for the top 20 features
plt.figure(figsize=(10, 6))
plt.barh(top_features, top_importances)
plt.title('Top 20 Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.gca().invert_yaxis()  # Ensure the most important feature is at the top
plt.tight_layout()
plt.show()

