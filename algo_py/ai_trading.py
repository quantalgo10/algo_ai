import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime
import plotly.graph_objects as go
from xgboost import XGBClassifier
import tensorflow as tf

st.title('Nifty Option Trading Signals with AI Models, Paper Trading, and Backtesting')

class AITrading:
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestClassifier(),
            'XGBoost': XGBClassifier(),
            'LSTM': None
        }
        
    def preprocess_data(self, data):
        """Preprocess data for ML models"""
        if data is None or data.empty:
            return None
            
        try:
            df = data.copy()
            
            # Calculate returns using Close price instead of Adj Close
            df['Daily Return'] = df['Close'].pct_change()
            
            # Add technical indicators
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['RSI'] = self.calculate_rsi(df['Close'])
            
            # Remove NaN values
            df = df.dropna()
            
            return df
            
        except Exception as e:
            st.error(f"Error preprocessing data: {str(e)}")
            return None
            
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    def render_dashboard(self):
        """Render AI Trading dashboard"""
        st.subheader("AI Trading Strategy Builder")
        
        # Strategy selection
        strategy = st.selectbox(
            "Select AI Strategy",
            ["Deep Learning", "Machine Learning", "Ensemble", "Reinforcement Learning"]
        )
        
        # Strategy specific settings
        if strategy == "Deep Learning":
            self._render_deep_learning()
        elif strategy == "Machine Learning":
            self._render_machine_learning()
        elif strategy == "Ensemble":
            self._render_ensemble()
        else:
            self._render_reinforcement_learning()
            
    def _render_deep_learning(self):
        st.subheader("Deep Learning Strategy")
        col1, col2 = st.columns(2)
        
        with col1:
            layers = st.number_input("Number of Layers", 1, 5, 2)
            neurons = st.number_input("Neurons per Layer", 16, 256, 64)
            
        with col2:
            dropout = st.slider("Dropout Rate", 0.0, 0.5, 0.2)
            epochs = st.number_input("Training Epochs", 10, 1000, 100)
            
        if st.button("Train Model"):
            st.info("Training deep learning model...")
            # Add model training code here
            
    def _render_machine_learning(self):
        st.subheader("Machine Learning Strategy")
        model_type = st.selectbox(
            "Select Model",
            list(self.models.keys())
        )
        
        # Model specific parameters
        self._render_model_params(model_type)
        
        if st.button("Train Model"):
            st.info(f"Training {model_type} model...")
            # Add model training code here
            
    def _render_ensemble(self):
        st.subheader("Ensemble Strategy")
        # Add ensemble strategy UI and logic
        
    def _render_reinforcement_learning(self):
        st.subheader("Reinforcement Learning Strategy")
        # Add RL strategy UI and logic
        
    def _render_model_params(self, model_type):
        if model_type == "Random Forest":
            st.number_input("Number of Trees", 100, 1000, 500)
            st.number_input("Max Depth", 3, 20, 10)
        elif model_type == "XGBoost":
            st.number_input("Number of Estimators", 100, 1000, 500)
            st.number_input("Learning Rate", 0.001, 0.1, 0.01)
        elif model_type == "LSTM":
            st.number_input("Sequence Length", 10, 100, 30)
            st.number_input("Hidden Units", 32, 256, 128)

# Define the Nifty 50 ticker and fetch historical data
ticker = '^NSEI'
data = yf.download(ticker, start='2020-01-01', end='2023-12-31')
data['Daily Return'] = data['Adj Close'].pct_change()

# Generate features
data['SMA_10'] = data['Adj Close'].rolling(window=10).mean()
data['SMA_50'] = data['Adj Close'].rolling(window=50).mean()
data['Volatility'] = data['Daily Return'].rolling(window=10).std()
data.dropna(inplace=True)

# Generate target variable
data['Target'] = np.where(data['Daily Return'].shift(-1) > 0, 1, 0)

# Select features and target
features = ['SMA_10', 'SMA_50', 'Volatility']
X = data[features]
y = data['Target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the AI models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression()
}

# Train and evaluate each model
results = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    results[model_name] = accuracy

# Display model accuracies
st.write("Model Accuracies")
st.write(results)

# Select the best model based on accuracy
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
st.write(f"Best Model: {best_model_name} with Accuracy: {results[best_model_name]:.2f}")

# Implement a simple trading strategy based on the best model's predictions
data['Prediction'] = best_model.predict(X)
data['Buy/Sell Signal'] = np.where(data['Prediction'] == 1, 'Buy', 'Sell')

# Paper trading and backtesting
initial_capital = 100000  # Initial capital for paper trading
data['Position'] = np.where(data['Buy/Sell Signal'] == 'Buy', 1, 0)
data['Portfolio Value'] = initial_capital * (1 + (data['Daily Return'] * data['Position']).cumsum())

# Display the buy/sell signals along with trading symbol, token, LTP, etc.
data['Trading Symbol'] = ticker
data['Token'] = 'NSE'
data['LTP'] = data['Adj Close']

# Display the results in a table
st.write("Trading Signals")
st.dataframe(data[['Trading Symbol', 'Token', 'LTP', 'Buy/Sell Signal', 'Portfolio Value']].tail())

# Plot the results
st.write("Cumulative Returns")
data['Strategy Return'] = data['Daily Return'] * data['Prediction'].shift(1)
data['Cumulative Market Return'] = (1 + data['Daily Return']).cumprod()
data['Cumulative Strategy Return'] = (1 + data['Strategy Return']).cumprod()

st.line_chart(data[['Cumulative Market Return', 'Cumulative Strategy Return']])