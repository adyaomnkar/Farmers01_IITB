import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

file_path = 'vprice.csv'
df = pd.read_csv(file_path)

df['Date'] = pd.to_datetime(df['Date'])

vegetables = df['Commodity'].unique()
print("Available vegetables:")
for i, veg in enumerate(vegetables, 1):
    print(f"{i}. {veg}")

veg_choice = int(input("Choose a vegetable by number: "))
selected_vegetable = vegetables[veg_choice - 1]

veg_data = df[df['Commodity'] == selected_vegetable].copy()

plt.figure(figsize=(10, 6))
sns.lineplot(data=veg_data, x='Date', y='Average', marker='o')
plt.title(f"Price Trend of {selected_vegetable}")
plt.ylabel('Average Price (in Kg)')
plt.xlabel('Date')
plt.xticks(rotation=45)

veg_data['Days'] = (veg_data['Date'] - veg_data['Date'].min()).dt.days

X = veg_data[['Days']]
y = veg_data['Average']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

future_day = np.array([[X['Days'].max() + 1]])
predicted_price = model.predict(future_day)

plt.text(0.95, 0.95, f"Predicted price: ₹{predicted_price[0]:.2f}", 
         horizontalalignment='right', verticalalignment='top', 
         transform=plt.gca().transAxes, fontsize=12, color='blue', 
         bbox=dict(facecolor='white', alpha=0.5))

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(data=veg_data[['Minimum', 'Maximum', 'Average']])
plt.title(f"Price Distribution of {selected_vegetable}")
plt.ylabel('Price (in Kg)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print(f"The predicted price for {selected_vegetable} for the next day is: ₹{predicted_price[0]:.2f}")
