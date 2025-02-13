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

# Encode non numerical data using get_dummies and drop the first level for each category
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

# Compute Adjusted R^2
n = X_test.shape[0]  # Number of observations
p = X_test.shape[1]  # Number of predictors
adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
print("Adjusted R^2 Score:", adjusted_r2)

# Define new car data for prediction
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
# Pick the first tree in the forest for visualization
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
