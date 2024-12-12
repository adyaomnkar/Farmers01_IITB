import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset from the provided text file
data = pd.read_csv('/Users/adyaomnkarpanda/Desktop/CODING/hackathon/farmer/algorithms/yield/yield_df.csv')  # Update with your actual path if needed

# Display the first few rows of the dataset
print("Initial Data:")
print(data.head())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Fill missing values or drop them (simple forward fill method)
data.fillna(method='ffill', inplace=True)

# Rename columns for easier access (if necessary)
data.rename(columns={'hg/ha_yield': 'yield'}, inplace=True)

# Split the data into features and target variable
X = data.drop('yield', axis=1)  # Features
y = data['yield']  # Target variable

# Convert categorical variables to dummy variables (if any)
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Predictions
rf_predictions = rf_model.predict(X_test)

# Evaluate model performance
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_rmse = rf_mse ** 0.5

print(f"\nRandom Forest Model Performance:")
print(f"Mean Absolute Error (MAE): {rf_mae}")
print(f"Mean Squared Error (MSE): {rf_mse}")
print(f"Root Mean Squared Error (RMSE): {rf_rmse}")

# Hyperparameter tuning using GridSearchCV (optional)
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20],
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid,
                           scoring='neg_mean_absolute_error', cv=5)
grid_search.fit(X_train, y_train)

best_rf_model = grid_search.best_estimator_
print("\nBest parameters from Grid Search:")
print(grid_search.best_params_)

# Predictions with the best model
best_rf_predictions = best_rf_model.predict(X_test)

# Evaluate best model performance
best_rf_mae = mean_absolute_error(y_test, best_rf_predictions)
best_rf_mse = mean_squared_error(y_test, best_rf_predictions)
best_rf_rmse = best_rf_mse ** 0.5

print(f"\nBest Random Forest Model Performance:")
print(f"Mean Absolute Error (MAE): {best_rf_mae}")
print(f"Mean Squared Error (MSE): {best_rf_mse}")
print(f"Root Mean Squared Error (RMSE): {best_rf_rmse}")
