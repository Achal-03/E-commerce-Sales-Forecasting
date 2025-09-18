import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import seaborn as sns

# Load the dataset
df = pd.read_csv('ahh.csv')

# Prepare data for LSTM
def prepare_data(data, look_back=3):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)

# Create LSTM model
def create_lstm_model(X_train):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Function to calculate custom accuracy for regression
def calculate_accuracy(y_true, y_pred, threshold=0.05):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_true_binary = np.ones_like(y_true)

    diff_percentage = np.abs(y_true - y_pred) / np.where(y_true != 0, y_true, 1)  # Avoid division by zero
    y_pred_binary = (diff_percentage <= threshold).astype(int)

    return accuracy_score(y_true_binary, y_pred_binary)

# Function to forecast future values
def forecast_future(model, scaler, last_sequence, n_days=7):
    last_sequence_scaled = scaler.transform(last_sequence.reshape(-1, 1))
    future_predictions = []

    current_sequence = last_sequence_scaled[-3:].reshape(1, 3, 1)  # Match LSTM input shape

    for _ in range(n_days):
        next_pred = model.predict(current_sequence, verbose=0)
        future_predictions.append(next_pred[0, 0])
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[0, -1, 0] = next_pred

    return scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Train and forecast function
def train_and_forecast(data, column_name):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))

    X, y = prepare_data(scaled_data)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = create_lstm_model(X_train)
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

    # Save model to desktop
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "saved_models")
    os.makedirs(desktop_path, exist_ok=True)
    sanitized_column_name = column_name.replace(" ", "_").lower()
    model_filename = f'model_{sanitized_column_name}.h5'
    model.save(os.path.join(desktop_path, model_filename))

    train_predict = scaler.inverse_transform(model.predict(X_train, verbose=0))
    test_predict = scaler.inverse_transform(model.predict(X_test, verbose=0))
    y_train_inv = scaler.inverse_transform(y_train)
    y_test_inv = scaler.inverse_transform(y_test)

    train_accuracy = calculate_accuracy(y_train_inv, train_predict)
    test_accuracy = calculate_accuracy(y_test_inv, test_predict)

    train_r2 = r2_score(y_train_inv, train_predict)
    test_r2 = r2_score(y_test_inv, test_predict)

    last_sequence = data[-3:]
    future_pred = forecast_future(model, scaler, last_sequence, n_days=7)

    return {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'future_predictions': future_pred
    }

# Initialize storage
accuracy_df = pd.DataFrame(columns=['Train Accuracy', 'Test Accuracy', 'Train R2', 'Test R2'])
forecast_df = pd.DataFrame()

print("\nTraining models and calculating accuracy scores...")

# Process overall sales
print("\nOverall Sales:")
overall_data = df['Overall Sales'].values
overall_results = train_and_forecast(overall_data, 'Overall Sales')

accuracy_df.loc['Overall Sales'] = [
    overall_results['train_accuracy'],
    overall_results['test_accuracy'],
    overall_results['train_r2'],
    overall_results['test_r2']
]
forecast_df['Overall_Sales_Forecast'] = overall_results['future_predictions'].flatten()

# Process individual products
product_columns = [col for col in df.columns if col.startswith('Product')]

for product in product_columns:
    print(f"\n{product}:")
    product_data = df[product].values
    product_results = train_and_forecast(product_data, product)

    accuracy_df.loc[product] = [
        product_results['train_accuracy'],
        product_results['test_accuracy'],
        product_results['train_r2'],
        product_results['test_r2']
    ]
    forecast_df[f'{product}_Forecast'] = product_results['future_predictions'].flatten()

# Save forecasts
forecast_path = 'forecasts'
os.makedirs(forecast_path, exist_ok=True)

forecast_filename = os.path.join(forecast_path, 'daily_sales_forecast.csv')
forecast_df.to_csv(forecast_filename)

accuracy_filename = os.path.join(forecast_path, 'model_accuracy_metrics.csv')
accuracy_df.to_csv(accuracy_filename)

# Visualizing Accuracy
plt.figure(figsize=(12, 6))
accuracy_df[['Train Accuracy', 'Test Accuracy']].plot(kind='bar')
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy Score (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(forecast_path, 'accuracy_comparison.png'))
plt.show()

# Visualizing Forecast
plt.figure(figsize=(12, 6))
forecast_df.plot(kind='bar')
plt.title('7-Day Sales Forecast')
plt.ylabel('Predicted Sales')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(forecast_path, 'sales_forecast.png'))
plt.show()

# Generate Monthly Forecast

forecast_months = pd.date_range(start=pd.to_datetime(df['Month'].iloc[-1]), periods=4, freq='M')[1:]
# monthly_forecast = forecast_df.sum(axis=1).groupby(pd.Grouper(freq='M')).sum()
# Generate a date range starting from the next day after the last date in the dataset
start_date = pd.to_datetime(df['Month'].iloc[-1]) + pd.DateOffset(days=1)
forecast_dates = pd.date_range(start=start_date, periods=len(forecast_df), freq='D')

# Assign DateTimeIndex to forecast_df
forecast_df.index = forecast_dates

# Resample to get monthly totals
monthly_forecast = forecast_df.resample('M').sum()

# Rename index to readable format (YYYY-MM)
monthly_forecast.index = monthly_forecast.index.strftime('%Y-%m')





# monthly_forecast.index = forecast_months.strftime('%Y-%m')
# Ensure forecast_df has a proper DateTimeIndex
forecast_df.index = pd.date_range(start=pd.to_datetime(df['Month'].iloc[-1]) + pd.Timedelta(days=1),
                                  periods=len(forecast_df), freq='D')

# Resample to get monthly totals
monthly_forecast = forecast_df.resample('M').sum()

# Generate expected forecast month labels
forecast_months = pd.date_range(start=forecast_df.index[0], periods=len(monthly_forecast), freq='M')

# Assign correct index
monthly_forecast.index = forecast_months.strftime('%Y-%m')





print("\nSales Forecast for Next 3 Months:")
print(monthly_forecast.round(2))

# Save Monthly Forecast
monthly_forecast.to_csv(os.path.join(forecast_path, 'monthly_sales_forecast.csv'))

# Visualizing Monthly Forecast
plt.figure(figsize=(12, 6))
monthly_forecast.plot(kind='bar', color='blue', alpha=0.7)
plt.title('3-Month Sales Forecast')
plt.ylabel('Predicted Sales')
plt.xlabel('Month')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(forecast_path, 'monthly_sales_forecast.png'))
plt.show()

# Redefine the train_and_forecast function with a 7-day forecast
def train_and_forecast(data, column_name):
    # Scale data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))

    # Prepare data
    X, y = prepare_data(scaled_data)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Create and train model
    model = create_lstm_model(X_train)
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

    # Save model
    model_dir = 'saved_models'
    os.makedirs(model_dir, exist_ok=True)
    sanitized_column_name = column_name.replace(" ", "_").lower()
    model_filename = f'model_{sanitized_column_name}.h5'
    model_path = os.path.join(model_dir, model_filename)
    model.save(model_path)
    print(f"Model for '{column_name}' saved to {model_path}")

    # Generate predictions
    train_predict = model.predict(X_train, verbose=0)
    test_predict = model.predict(X_test, verbose=0)

    # Inverse transform predictions and actual values
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    y_train_inv = scaler.inverse_transform(y_train)
    y_test_inv = scaler.inverse_transform(y_test)

    # Calculate accuracy
    train_accuracy = calculate_accuracy(y_train_inv, train_predict)
    test_accuracy = calculate_accuracy(y_test_inv, test_predict)

    # Calculate R2 score
    train_r2 = r2_score(y_train_inv, train_predict)
    test_r2 = r2_score(y_test_inv, test_predict)

    # Forecast next 7 days
    last_sequence = data[-7:]
    future_pred = forecast_future(model, scaler, last_sequence, n_days=7)

    return {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'future_predictions': future_pred,
        'model': model,
        'scaler': scaler
    }

# Function to forecast future values (Modified for 7 Days)
def forecast_future(model, scaler, last_sequence, n_days=7):
    last_sequence_scaled = scaler.transform(last_sequence.reshape(-1, 1))
    future_predictions = []

    current_sequence = last_sequence_scaled[-3:].reshape(1, 3, 1)

    for _ in range(n_days):
        next_pred = model.predict(current_sequence, verbose=0)
        future_predictions.append(next_pred[0, 0])
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[0, -1, 0] = next_pred

    return scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Create forecasts and calculate accuracy for all products
results = {}
forecast_df = pd.DataFrame()
accuracy_df = pd.DataFrame(columns=['Train Accuracy', 'Test Accuracy', 'Train R2', 'Test R2'])

print("\nTraining models and calculating accuracy scores...")

# First, process overall sales
print("\nOverall Sales:")
overall_data = df['Overall Sales'].values
overall_results = train_and_forecast(overall_data, 'Overall Sales')

accuracy_df.loc['Overall Sales'] = [
    overall_results['train_accuracy'],
    overall_results['test_accuracy'],
    overall_results['train_r2'],
    overall_results['test_r2']
]
forecast_df['Overall_Sales_Forecast'] = overall_results['future_predictions'].flatten()

# Process individual products
product_columns = [col for col in df.columns if col.startswith('Product')]

for product in product_columns:
    print(f"\n{product}:")
    product_data = df[product].values
    product_results = train_and_forecast(product_data, product)

    accuracy_df.loc[product] = [
        product_results['train_accuracy'],
        product_results['test_accuracy'],
        product_results['train_r2'],
        product_results['test_r2']
    ]
    forecast_df[f'{product}_Forecast'] = product_results['future_predictions'].flatten()

# Create forecast table for next 7 days
last_date = pd.to_datetime(df['Month'].iloc[-1])
forecast_days = pd.date_range(start=last_date + pd.Timedelta(days=1),
                              periods=7, freq='D')
forecast_df.index = forecast_days.strftime('%Y-%m-%d')

# Save forecasts to CSV
forecast_path = 'forecasts'
os.makedirs(forecast_path, exist_ok=True)

# Save daily forecasts
forecast_filename = os.path.join(forecast_path, 'daily_sales_forecast.csv')
forecast_df.to_csv(forecast_filename)
print(f"\nDaily forecasts saved to {forecast_filename}")

# Save accuracy metrics
accuracy_filename = os.path.join(forecast_path, 'model_accuracy_metrics.csv')
accuracy_df.to_csv(accuracy_filename)
print(f"Accuracy metrics saved to {accuracy_filename}")

# Display results
print("\nModel Accuracy Scores:")
print(accuracy_df.round(4) * 100)

print("\nAverage Model Performance:")
avg_metrics = accuracy_df.mean() * 100
print(avg_metrics.round(2))

print("\nSales Forecast for Next 7 Days:")
print(forecast_df.round(2))

# Visualizations
viz_path = os.path.join(forecast_path, 'visualizations')
os.makedirs(viz_path, exist_ok=True)

# Accuracy scores visualization
plt.figure(figsize=(12, 6))
accuracy_df[['Train Accuracy', 'Test Accuracy']].plot(kind='bar')
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy Score (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(viz_path, 'accuracy_comparison.png'))
plt.show()

# Forecast visualization
plt.figure(figsize=(12, 6))
forecast_df.plot(kind='bar')
plt.title('7-Day Sales Forecast')
plt.ylabel('Predicted Sales')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(viz_path, 'sales_forecast.png'))
plt.show()

from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Ensure 'y_train' is a Pandas Series (Time Series Data)
y_train = pd.Series(y_train)  # Convert to Series if it's not already

# Train the Holt-Winters Model
model = ExponentialSmoothing(y_train, seasonal='add', seasonal_periods=12).fit()

# After creating your forecast DataFrame, add these lines:

from google.colab import files

# Save the forecast to a CSV file
forecast_filename = 'three_day_sales_forecast.csv'
daily_forecast_df.to_csv(forecast_filename)

# Download the file
files.download(forecast_filename)

print(f"\nDownloading forecast file: {forecast_filename}")

import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import seaborn as sns
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.optimizers import AdamW


# Load the dataset
df = pd.read_csv('ahh.csv')

# Prepare data for LSTM
def prepare_data(data, look_back=3):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)

# Create LSTM model
def create_lstm_model(X_train):
    model = Sequential([
        # CNN Feature Extraction
        Conv1D(filters=128, kernel_size=5, activation='relu', padding='same', input_shape=(X_train.shape[1], X_train.shape[2])),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),

        # Bidirectional LSTM for better sequence learning
        Bidirectional(LSTM(100, return_sequences=True)),
        Dropout(0.2),

        LSTM(100),
        Dropout(0.2),

        # Fully Connected Layer
        Dense(50, activation='relu'),
        Dropout(0.2),

        Dense(1)  # Output layer
    ])
    optimizer = AdamW(learning_rate=0.001, weight_decay=1e-5)

    model.compile(optimizer='adam', loss='mse')
    return model

# Function to calculate custom accuracy for regression
def calculate_accuracy(y_true, y_pred, threshold=0.05):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_true_binary = np.ones_like(y_true)

    diff_percentage = np.abs(y_true - y_pred) / np.where(y_true != 0, y_true, 1)  # Avoid division by zero
    y_pred_binary = (diff_percentage <= threshold).astype(int)

    return accuracy_score(y_true_binary, y_pred_binary)

# Function to forecast future values
def forecast_future(model, scaler, last_sequence, n_days=7):
    last_sequence_scaled = scaler.transform(last_sequence.reshape(-1, 1))
    future_predictions = []

    current_sequence = last_sequence_scaled[-3:].reshape(1, 3, 1)  # Match LSTM input shape

    for _ in range(n_days):
        next_pred = model.predict(current_sequence, verbose=0)
        future_predictions.append(next_pred[0, 0])
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[0, -1, 0] = next_pred

    return scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))


from google.colab import drive
import os

drive.flush_and_unmount()


# Mount Google Drive
# drive.mount('/content/drive')
drive.mount('/content/my_drive')


# Define Drive path for saving models
drive_path = "/content/my_drive/MyDrive/sales_forecasting"
os.makedirs(drive_path, exist_ok=True)  # Ensure directory exists

def train_and_forecast(data, column_name):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))

    X, y = prepare_data(scaled_data)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = create_lstm_model(X_train)
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

    # ✅ Save Model to Google Drive
    sanitized_column_name = column_name.replace(" ", "_").lower()
    model_filename = f'model_{sanitized_column_name}.h5'
    model.save(os.path.join(drive_path, model_filename))  # Save to Drive

    train_predict = scaler.inverse_transform(model.predict(X_train, verbose=0))
    test_predict = scaler.inverse_transform(model.predict(X_test, verbose=0))
    y_train_inv = scaler.inverse_transform(y_train)
    y_test_inv = scaler.inverse_transform(y_test)

    train_accuracy = calculate_accuracy(y_train_inv, train_predict)
    test_accuracy = calculate_accuracy(y_test_inv, test_predict)

    train_r2 = r2_score(y_train_inv, train_predict)
    test_r2 = r2_score(y_test_inv, test_predict)

    last_sequence = data[-3:]
    future_pred = forecast_future(model, scaler, last_sequence, n_days=7)

    return {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'future_predictions': future_pred
    }


# Initialize storage
accuracy_df = pd.DataFrame(columns=['Train Accuracy', 'Test Accuracy', 'Train R2', 'Test R2'])
forecast_df = pd.DataFrame()

print("\nTraining models and calculating accuracy scores...")

# Process overall sales
print("\nOverall Sales:")
overall_data = df['Overall Sales'].values
overall_results = train_and_forecast(overall_data, 'Overall Sales')

accuracy_df.loc['Overall Sales'] = [
    overall_results['train_accuracy'],
    overall_results['test_accuracy'],
    overall_results['train_r2'],
    overall_results['test_r2']
]
forecast_df['Overall_Sales_Forecast'] = overall_results['future_predictions'].flatten()

# Process individual products
product_columns = [col for col in df.columns if col.startswith('Product')]

for product in product_columns:
    print(f"\n{product}:")
    product_data = df[product].values
    product_results = train_and_forecast(product_data, product)

    accuracy_df.loc[product] = [
        product_results['train_accuracy'],
        product_results['test_accuracy'],
        product_results['train_r2'],
        product_results['test_r2']
    ]
    forecast_df[f'{product}_Forecast'] = product_results['future_predictions'].flatten()

# Save forecasts
forecast_path = 'forecasts'
os.makedirs(forecast_path, exist_ok=True)

forecast_filename = os.path.join(forecast_path, 'daily_sales_forecast.csv')
forecast_df.to_csv(forecast_filename)

accuracy_filename = os.path.join(forecast_path, 'model_accuracy_metrics.csv')
accuracy_df.to_csv(accuracy_filename)

# Visualizing Accuracy
plt.figure(figsize=(12, 6))
accuracy_df[['Train Accuracy', 'Test Accuracy']].plot(kind='bar')
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy Score (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(forecast_path, 'accuracy_comparison.png'))
plt.show()

# Visualizing Forecast
plt.figure(figsize=(12, 6))
forecast_df.plot(kind='bar')
plt.title('7-Day Sales Forecast')
plt.ylabel('Predicted Sales')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(forecast_path, 'sales_forecast.png'))
plt.show()

# Save forecasts to Google Drive
forecast_filename = os.path.join(drive_path, 'daily_sales_forecast.csv')
forecast_df.to_csv(forecast_filename)

accuracy_filename = os.path.join(drive_path, 'model_accuracy_metrics.csv')
accuracy_df.to_csv(accuracy_filename)

print(f"Forecast data saved successfully at {forecast_filename}")
print(f"Accuracy metrics saved successfully at {accuracy_filename}")

overall_sales = os.path.join(drive_path, 'Overall_Sales.csv')
df.to_csv(overall_sales)
