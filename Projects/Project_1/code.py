# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Load Excel file and check sheets
excel_file = pd.ExcelFile("online_retail_II.xlsx")
print("Sheet names:", excel_file.sheet_names)

# Load the data from the required sheet
df = pd.read_excel("online_retail_II.xlsx", sheet_name="Year 2010-2011")

# Display basic info
print("Initial Data Shape:", df.shape)
print(df.head())

# Drop rows with missing Customer ID
df.dropna(subset=["Customer ID"], inplace=True)

# Rename columns for consistency
df.rename(columns={
    'Customer ID': 'CustomerID',
    'Invoice': 'InvoiceNo',
    'Price': 'UnitPrice'
}, inplace=True)

# Add TotalPrice column
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

# Set snapshot date for Recency calculation (next day after last invoice)
snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

# Aggregate data per customer
customer_df = df.groupby("CustomerID").agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,  # Recency
    'InvoiceNo': 'nunique',                                   # Frequency
    'TotalPrice': 'sum',                                      # Monetary
    'Quantity': 'sum',                                        # Total quantity
    'UnitPrice': 'mean',                                      # Avg unit price
    'Country': 'first'                                        # Country
}).reset_index()

# Rename columns
customer_df.rename(columns={
    'InvoiceDate': 'Recency',
    'InvoiceNo': 'Frequency',
    'TotalPrice': 'Monetary',
    'Quantity': 'TotalQuantity',
    'UnitPrice': 'AvgUnitPrice'
}, inplace=True)

# Display the aggregated customer data
print("Customer Data Sample:")
print(customer_df.head())

# Filter out extreme or invalid values
filtered_df = customer_df[(customer_df['Monetary'] > 0) & 
                          (customer_df['Frequency'] > 0) & 
                          (customer_df['Recency'] >= 0)]

# Create features (X) and target (y)
X = filtered_df[['Recency', 'Frequency', 'Monetary']]
y = filtered_df['Monetary'] * filtered_df['Frequency']  # Proxy CLV

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("R² Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# Feature importance
feature_importance = pd.Series(model.feature_importances_, index=['Recency', 'Frequency', 'Monetary'])
feature_importance.sort_values().plot(kind='barh', title='Feature Importance')
plt.tight_layout()
plt.show()

# Plotting scatterplot
plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual CLV (Proxy)")
plt.ylabel("Predicted CLV")
plt.title("Actual vs Predicted Customer Lifetime Value")
plt.grid(True)
plt.tight_layout()
plt.show()
