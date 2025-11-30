# train_model.py

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load and clean data
df = pd.read_csv("car_dataset.csv")
df.dropna(inplace=True)
df['Car_Age'] = 2025 - df['Year']
df.drop(['Car_Name', 'Year'], axis=1, inplace=True)

# Feature engineering
df['Price_per_KM'] = df['Selling_Price'] / df['Kms_Driven']
df['Age_Category'] = pd.cut(df['Car_Age'], bins=[0,3,7,100], labels=['New','Mid','Old'])

# Encoding
le = LabelEncoder()
df['Age_Category'] = le.fit_transform(df['Age_Category'])
df = pd.get_dummies(df, columns=['Fuel_Type', 'Transmission', 'Seller_Type'], drop_first=True)

# Scaling
scaler = StandardScaler()
num_cols = ['Car_Age', 'Kms_Driven', 'Present_Price', 'Price_per_KM']
df[num_cols] = scaler.fit_transform(df[num_cols])

# Train model
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Save model
# Save model and test data
with open("random_forest_model.pkl", "wb") as f:
    pickle.dump((model, X_test, y_test), f)

print("✅ Model and test data saved as random_forest_model.pkl")

