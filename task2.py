import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import os
# Load the dataset
data = pd.read_csv("house_prices.csv")
# Display dataset preview and basic info
print("Dataset Preview:")
print(data.head())
print("\nDataset Info:")
print(data.info())
# Data Preprocessing
# Convert categorical variables (waterfront, condition) to numeric
categorical_mappings = {
    'waterfront': {'N': 0, 'Y': 1},
    'condition': {'Poor': 1, 'Fair': 2, 'Good': 3, 'Very Good': 4, 'Excellent': 5}
}
for column, mapping in categorical_mappings.items():
    if column in data.columns:
        data[column] = data[column].map(mapping)
# Handle missing values (if any)
# Ensure non-numeric columns are excluded when calculating the median
data_numeric = data.select_dtypes(include=[np.number])
data[data_numeric.columns] = data_numeric.fillna(data_numeric.median())
# Separate features and target variable
X = data.drop(columns=['price', 'id', 'date'])  # Exclude non-predictive columns
y = data['price']
# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Feature Importance using Random Forest
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
# Plot Feature Importance
importance = rf_model.feature_importances_
sns.barplot(x=importance, y=X.columns)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()
# Train Models
# 1. Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_lr = linear_model.predict(X_test)
# 2. Random Forest
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
# Evaluate Models
print("Linear Regression Metrics:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_lr))
print("R2 Score:", r2_score(y_test, y_pred_lr))
print("\nRandom Forest Metrics:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_rf))
print("R2 Score:", r2_score(y_test, y_pred_rf))
# Save Predictions to CSV
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder = "/mnt/data/output"
os.makedirs(output_folder, exist_ok=True)
output_file_path = os.path.join(output_folder, f"predictions_{current_time}.csv")
predictions = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price (Linear)": y_pred_lr,
    "Predicted Price (Random Forest)": y_pred_rf
})
predictions.to_csv(output_file_path, index=False)
print(f"Predictions saved to '{output_file_path}'.")
# Plot Actual vs Predicted
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred_lr, label="Linear Regression", alpha=0.7)
plt.scatter(y_test, y_pred_rf, label="Random Forest", alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", color="red")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.legend()
plt.title("Actual vs Predicted Prices")
plt.show()
