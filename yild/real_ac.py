import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the data from the CSV file
data = pd.read_csv('/Users/adyaomnkarpanda/Desktop/yild/_crop+yield+prediction_data_crop_yield.csv')

# Preprocess the data
X = data.drop(['Crop', 'Yield'], axis=1)
y = data['Yield']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test_scaled)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('Mean Squared Error:', mse)
print('R-squared:', r2)

# User-interactive prediction
def predict_yield():
    crop = input('Enter the crop name (Cocoa, beans, Oil palm fruit, Rice, paddy, Rubber, natural): ')
    precipitation = float(input('Enter precipitation (mm/day): '))
    specific_humidity = float(input('Enter specific humidity (g/kg): '))
    relative_humidity = float(input('Enter relative humidity (%): '))
    temperature = float(input('Enter temperature (C): '))

    # Create a new data point
    new_data = [[precipitation, specific_humidity, relative_humidity, temperature]]
    new_data_scaled = scaler.transform(new_data)

    # Predict the yield
    predicted_yield = model.predict(new_data_scaled)

    print(f'Predicted yield for {crop}: {predicted_yield[0]}')

# Run the interactive prediction
predict_yield()