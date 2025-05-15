
# Stock Price Prediction using AI (Time Series Analysis)
# Author: Poornachandran M
# Tools: Python, Yahoo Finance, Linear Regression

# Step 1: Install yfinance if not available
# !pip install yfinance

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Step 2: Load stock data
stock_symbol = "INFY.NS"  # Change this to any stock like 'TSLA', 'TCS.NS', etc.
data = yf.download(stock_symbol, start="2020-01-01", end="2024-12-31")
data = data[["Close"]]
data.dropna(inplace=True)
data["Days"] = np.arange(len(data))

# Step 3: Split data into X (days) and y (prices)
X = data["Days"].values.reshape(-1, 1)
y = data["Close"].values.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 4: Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predict next 30 days
future_days = np.arange(len(data), len(data)+30).reshape(-1, 1)
future_pred = model.predict(future_days)

# Step 6: Plot the data
plt.figure(figsize=(12,6))
plt.plot(data["Close"], label="Historical Prices")
plt.plot(np.arange(len(data), len(data)+30), future_pred, label="Predicted Prices", linestyle='dashed')
plt.xlabel("Days")
plt.ylabel("Stock Price (INR)")
plt.title(f"{stock_symbol} Stock Price Prediction")
plt.legend()
plt.grid(True)
plt.show()
