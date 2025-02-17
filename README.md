# Weather-Prediction-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Step 1: Load the Dataset
file_path = "/kaggle/input/weather-prediction/weather_prediction_dataset.csv"  # Replace with the actual path
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(data.head())

# Display column names to verify the target column
print("Column Names:", data.columns)

# Step 2: Data Preprocessing
# Handle missing values
data = data.dropna()  # or use data.fillna() to fill missing values

# Encode categorical variables (if any)
categorical_cols = data.select_dtypes(include=['object']).columns
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Separate features (X) and target (y)
# Replace 'BASEL_temp_mean' with your actual target column name
target_column = 'BASEL_temp_mean'  # Change this to your actual target column name
if target_column not in data.columns:
    raise ValueError(f"Target column '{target_column}' not found in the dataset. Available columns: {data.columns}")

X = data.drop(columns=[target_column])  # Features
y = data[target_column]  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize/Standardize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 3: Feature Extraction (if needed)
# You can create new features or select important ones using techniques like PCA, feature importance, etc.

# Step 4: Model Selection and Training
# Option 1: Machine Learning Model (Random Forest)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the model
y_pred_rf = rf_model.predict(X_test)
print("Random Forest RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))
print("Random Forest R2 Score:", r2_score(y_test, y_pred_rf))

# Option 2: Deep Learning Model (Neural Network)
dl_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)  # Output layer for regression
])

dl_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
dl_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model
y_pred_dl = dl_model.predict(X_test)
print("Deep Learning RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_dl)))
print("Deep Learning R2 Score:", r2_score(y_test, y_pred_dl))
