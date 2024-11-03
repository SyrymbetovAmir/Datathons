import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Function to load data
def load_data(file_path):
    try:
        return pd.read_csv(file_path, nrows=200000)
    except FileNotFoundError:
        print(f"Error: file {file_path} not found.")
        return None

# Load data
data = load_data('train.csv')
if data is None:
    exit()

# Define the target column
target_column = '6~7_ride'

# Prepare data
X = data.drop(columns=[target_column, 'id', 'date', 'station_name', 'latitude', 'longitude'])
y = data[target_column]

# Output unique values for each column
for column in X.columns:
    unique_values = X[column].unique()
    print(f"Unique values in '{column}': \n {unique_values}")

# Handle categorical variables and convert to numeric
X = pd.get_dummies(X, drop_first=True)  # Encode categorical variables
X = X.apply(pd.to_numeric, errors='coerce')  # Convert to numeric
X.dropna(inplace=True)  # Remove rows with NaN
y = y[X.index]  # Update y to match X

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Output results
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")
