Practical no 1Decompose time series data to find trend, seasonality, cyclic and irregularity.

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

# Load dataset
data = pd.read_csv("AirPassengers.csv", parse_dates=["Month"], index_col="Month")

# Plot original series
plt.plot(data.index, data["#Passengers"], color="green")
plt.title("Air Passengers Time Series")
plt.xlabel("Year")
plt.ylabel("Passengers")
plt.show()

# STL Decomposition
stl = STL(data["#Passengers"], seasonal=13)
result = stl.fit()

# Plot components
fig, axes = plt.subplots(3, 1, figsize=(7,4))

axes[0].plot(result.trend, color="red")
axes[0].set_title("Trend")

axes[1].plot(result.seasonal, color="blue")
axes[1].set_title("Seasonal")

axes[2].plot(result.resid)
axes[2].set_title("Residual")

plt.tight_layout()
plt.show()

practical no 2 Data conversion of non-stationary to stationary

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import boxcox

# Generate random walk time series
np.random.seed(42)
ts = pd.Series(np.cumsum(np.random.normal(size=100)))

def plot_series(series, title):
    plt.figure(figsize=(8,3))
    plt.plot(series)
    plt.title(title)
    plt.show()

# Original series
plot_series(ts, "Original Time Series")

# Differencing
diff = ts.diff().dropna()
plot_series(diff, "Differenced Series")

# Log transformation
log_ts = np.log(ts - ts.min() + 1)
plot_series(log_ts, "Log Transformation")

# Moving average removal
ma = ts.rolling(5).mean()
ma_diff = ts - ma
plot_series(ma_diff, "Moving Average Difference")

# Decomposition
decomp = seasonal_decompose(ts, model="additive", period=10)
residual = decomp.resid.dropna()

plt.figure(figsize=(8,4))
plt.subplot(311); plt.plot(decomp.trend); plt.title("Trend")
plt.subplot(312); plt.plot(decomp.seasonal); plt.title("Seasonal")
plt.subplot(313); plt.plot(decomp.resid); plt.title("Residual")
plt.tight_layout(); plt.show()

# Box-Cox transformation
boxcox_ts, lam = boxcox(ts - ts.min() + 1)
plot_series(boxcox_ts, f"Box-Cox Transformation (λ={lam:.2f})")

# ADF Test function
def adf_test(series, name):
    result = adfuller(series.dropna())
    print(f"\nADF Test: {name}")
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])
    print("Stationary" if result[1] <= 0.05 else "Not Stationary")

# Run tests
adf_test(ts, "Original")
adf_test(diff, "Differenced")
adf_test(log_ts, "Log")
adf_test(ma_diff, "Moving Avg Diff")
adf_test(residual, "Residual")
adf_test(pd.Series(boxcox_ts), "Box-Cox")

Practical no 3 Perform a duckey-fuller test to check stationarity of data

import pandas as pd
from statsmodels.tsa.stattools import adfuller

# Create example time series data
dates = pd.date_range("2020-01-01", periods=100, freq="D")
values = [x + 0.1*x for x in range(100)]

df = pd.DataFrame({"date": dates, "value": values})
print(df.head())

# ADF Test
adf_result = adfuller(df["value"])

print("\nADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])

print("\nCritical Values:")
for key, val in adf_result[4].items():
    print(f"{key}: {val}")

# Interpretation
print("\nSeries is Stationary" if adf_result[1] <= 0.05 else "\nSeries is Not Stationary")

Practical no 4Implementation of moving averages models

import pandas as pd
import matplotlib.pyplot as plt

dates = pd.date_range("2020-01-01", "2020-01-31")
prices = [43,31,1,10,20,24,26,27,34,35,36,37,31,21,20,19,18,19,20,24,28,29,
          20,32,34,35,36,30,32,30,23]

df = pd.DataFrame({"Date": dates, "Price": prices})

# Moving averages
for window in [3,4,5]:
    df[f"{window}-MA"] = df["Price"].rolling(window).mean()

# Plot
df.plot(x="Date", y=["Price","3-MA","4-MA","5-MA"])
plt.show()

Practical no 5 Demonstration of autocorrelation functions and partial autocorrelation functions.

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf

# Load dataset
df = pd.read_csv("AirPassengers.csv")

# Plot time series
plt.figure(figsize=(7,4))
plt.plot(df['#Passengers'], color='green', label='Passengers')
plt.title("Air Passengers Dataset")
plt.xlabel("Time")
plt.ylabel("Passengers")
plt.legend()
plt.show()

# ACF and PACF plots
plot_acf(df['#Passengers'], lags=40)
plt.title("ACF")
plt.show()

plot_pacf(df['#Passengers'], lags=40, method='ywm')
plt.title("PACF")
plt.show()

# Numerical values
acf_values = acf(df['#Passengers'], nlags=40)
pacf_values = pacf(df['#Passengers'], nlags=40, method='ywm')

print("ACF Values:\n", acf_values)
print("\nPACF Values:\n", pacf_values)

Practical no 6 Implementation of Autoregressive models.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

# Generate synthetic time series
np.random.seed(20)
series = pd.Series([50 + 0.8*i + np.random.normal(scale=5) for i in range(100)])

# Train-Test split
train_size = int(len(series)*0.8)
train, test = series[:train_size], series[train_size:]

# Train AR model
model = AutoReg(train, lags=5).fit()

# Predictions
pred = model.predict(start=len(train), end=len(series)-1)

# Evaluation
mse = mean_squared_error(test, pred)
print("Mean Squared Error:", round(mse,4))

# Plot
plt.plot(test, label="Actual")
plt.plot(pred, "--", label="Predicted")
plt.title("AutoReg Model Prediction")
plt.legend()
plt.show()

Practical no 7Implementation of ARIMA model.

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# Load dataset
df = pd.read_csv("AirPassengers.csv")
series = df["#Passengers"]

# Plot original series
plt.plot(series)
plt.title("AirPassengers Time Series")
plt.show()

# ADF Test
def adf_test(data):
    result = adfuller(data.dropna())
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])

print("ADF Test for Original Series")
adf_test(series)

# Differencing
series_diff = series.diff().dropna()

print("\nADF Test After Differencing")
adf_test(series_diff)

# ACF & PACF
plot_acf(series_diff)
plt.show()

plot_pacf(series_diff, method="ywm")
plt.show()

# ARIMA Model
model = ARIMA(series, order=(1,1,1))
model_fit = model.fit()

print(model_fit.summary())

# Forecast
forecast = model_fit.forecast(steps=10)

plt.plot(series, label="Original")
plt.plot(range(len(series), len(series)+10), forecast, color="red", label="Forecast")
plt.legend()
plt.title("ARIMA Forecast")
plt.show()

Practical no 8 SARIMA model

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load dataset
df = pd.read_csv("AirPassengers.csv")
series = df["#Passengers"]

# Plot series
plt.plot(series)
plt.title("AirPassengers Dataset")
plt.show()

# ADF Test
def adf_test(data):
    result = adfuller(data.dropna())
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])

print("ADF Test")
adf_test(series)

# Fit SARIMAX Model
model = SARIMAX(series,
                order=(1,1,1),
                seasonal_order=(1,1,1,12))

model_fit = model.fit()

print(model_fit.summary())

# Forecast
forecast = model_fit.forecast(steps=12)

plt.plot(series, label="Original")
plt.plot(range(len(series), len(series)+12), forecast, color="red", label="Forecast")
plt.legend()
plt.title("SARIMAX Forecast")
plt.show()

Practical 9

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import SimpleExpSmoothing, Holt, ExponentialSmoothing

# Load dataset
data = pd.read_csv("AirPassengers.cwsv", parse_dates=["Month"], index_col="Month")

# Plot original data
plt.plot(data)
plt.title("Air Passengers Dataset")
plt.xlabel("Month")
plt.ylabel("Passengers")
plt.show()

# Function to train model and plot results
def plot_forecast(model, title):
    fit = model.fit()
    forecast = fit.forecast(40)

    plt.plot(data, label="Original")
    plt.plot(fit.fittedvalues, label="Fitted")
    plt.plot(forecast, label="Forecast")
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel("Passengers")
    plt.legend()
    plt.show()

# Single Exponential Smoothing
plot_forecast(SimpleExpSmoothing(data), "Single Exponential Smoothing")

# Double Exponential Smoothing (Holt)
plot_forecast(Holt(data), "Double Exponential Smoothing")

# Holt-Winters Method
plot_forecast(
    ExponentialSmoothing(data, trend="add", seasonal="add", seasonal_periods=12),
    "Holt-Winters Exponential Smoothing"
)
