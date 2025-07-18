# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Step 1: Generate Custom Traffic Dataset
def generate_traffic_data(num_samples=1000):
    """
    Generate a synthetic traffic dataset.
    """
    np.random.seed(42)
    
    # Features
    time_of_day = np.random.randint(0, 24, num_samples)  # Hour of the day (0-23)
    day_of_week = np.random.randint(0, 7, num_samples)   # Day of the week (0-6)
    weather_condition = np.random.choice(['sunny', 'rainy', 'cloudy'], num_samples)  # Weather condition
    road_type = np.random.choice(['highway', 'urban', 'rural'], num_samples)  # Type of road
    
    # Target: Traffic Volume (simulated based on features)
    traffic_volume = (
        500 * (time_of_day >= 7) * (time_of_day <= 9) +  # Morning rush hour
        700 * (time_of_day >= 16) * (time_of_day <= 19) +  # Evening rush hour
        300 * (day_of_week >= 5) +  # Weekend traffic
        200 * (weather_condition == 'rainy') +  # More traffic in rainy weather
        100 * (road_type == 'highway') +  # Highways have more traffic
        np.random.normal(0, 50, num_samples)  # Random noise
    )
    
    # Create DataFrame
    data = pd.DataFrame({
        'time_of_day': time_of_day,
        'day_of_week': day_of_week,
        'weather_condition': weather_condition,
        'road_type': road_type,
        'traffic_volume': traffic_volume
    })
    
    return data

# Step 2: Preprocess Data
def preprocess_data(data):
    """
    Preprocess the data: encode categorical variables and split into features and target.
    """
    # Encode categorical variables
    data = pd.get_dummies(data, columns=['weather_condition', 'road_type'], drop_first=True)
    
    # Split into features (X) and target (y)
    X = data.drop('traffic_volume', axis=1)
    y = data['traffic_volume']
    
    return X, y

# Step 3: Feature Engineering
def feature_engineering(X):
    """
    Standardize features.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

# Step 4: Train-Test Split
def split_data(X, y):
    """
    Split the data into training and testing sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Step 5: Machine Learning Model Training
def train_random_forest(X_train, y_train):
    """
    Train a Random Forest Regressor model.
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_neural_network(X_train, y_train):
    """
    Train a Neural Network model using TensorFlow/Keras.
    """
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)  # Output layer for regression
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
    return model

# Step 6: Model Evaluation
def evaluate_model(model, X_test, y_test, model_type='random_forest'):
    """
    Evaluate the model on the test set.
    """
    if model_type == 'random_forest':
        y_pred = model.predict(X_test)
    elif model_type == 'neural_network':
        y_pred = model.predict(X_test).flatten()
    
    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"Model: {model_type}")
    print(f"Mean Absolute Error: {mae}")
    print(f"Root Mean Squared Error: {rmse}")
    
    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Actual Traffic Volume")
    plt.ylabel("Predicted Traffic Volume")
    plt.title(f"Actual vs Predicted Traffic Volume ({model_type})")
    plt.show()

# Step 7: Main Function
def main():
    # Generate synthetic traffic data
    print("Generating synthetic traffic data...")
    data = generate_traffic_data(num_samples=1000)
    print(data.head())
    
    # Preprocess data
    X, y = preprocess_data(data)
    
    # Feature engineering
    X_scaled = feature_engineering(X)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)
    
    # Train Random Forest model
    print("Training Random Forest Model...")
    rf_model = train_random_forest(X_train, y_train)
    evaluate_model(rf_model, X_test, y_test, model_type='random_forest')
    
    # Train Neural Network model
    print("Training Neural Network Model...")
    nn_model = train_neural_network(X_train, y_train)
    evaluate_model(nn_model, X_test, y_test, model_type='neural_network')

if __name__ == "__main__":
    main()
