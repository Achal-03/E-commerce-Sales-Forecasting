E-commerce Sales Forecasting
This project provides an interactive dashboard and machine learning models for forecasting sales in an e-commerce setting. It combines a user-friendly Streamlit interface with advanced time series forecasting using LSTM neural networks and other techniques.

Features
Interactive Dashboard: Built with Streamlit, allowing users to log in and view sales analytics, forecasts, and product performance metrics.
Sales Forecasting: Uses LSTM and hybrid deep learning models to predict future sales for overall and individual products.
Data Visualization: Includes historical vs. forecasted sales plots, product-wise analysis, and performance metrics.
Automated Data Loading: Sales and forecast data are loaded automatically from Google Drive.
Model Training & Evaluation: Trains models on historical data, evaluates accuracy, and saves results and forecasts.

How It Works
Data Loading: The dashboard loads historical and forecasted sales data from Google Drive.
User Login: Users must log in to access the dashboard.
Visualization: The dashboard displays sales trends, product analysis, and performance metrics.
Modeling: The backend (current.py) trains LSTM models on sales data and generates forecasts for the next 7 days and 3 months.
Results: Forecasts and model accuracy metrics are saved as CSV files and visualized in the dashboard.

Requirements
Python 3.7+
Streamlit
Pandas
NumPy
Plotly
scikit-learn
TensorFlow / Keras
Matplotlib
Seaborn
gdown

Install dependencies with:
pip install streamlit pandas numpy plotly scikit-learn tensorflow matplotlib seaborn gdown

Usage
1. Run the Dashboard
streamlit run app.py

2. Train Models & Generate Forecasts
Edit and run current.py to process your dataset (ahh.csv), train models, and generate forecasts.

3. View Results
Forecasts and metrics are saved in the forecasts/ directory.
Visualizations are saved as PNG files for accuracy and forecast comparisons.
Customization
Update Google Drive file IDs in streamlit.py to use your own datasets.
Modify model parameters in current.py for different forecasting strategies.

This project is for educational and research purposes.
