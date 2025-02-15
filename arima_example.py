# %% ARIMA Example
# # following statsmodel example: https://www.statsmodels.org/stable/examples/notebooks/generated/tsa_arma_0.html#Exercise:-Can-you-obtain-a-better-fit-for-the-Sunspots-model?-(Hint:-sm.tsa.AR-has-a-method-select_order)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.api import qqplot
from statsmodels.tsa.ar_model import AutoReg

import jax
import jax.numpy as jnp


# %%
dta = sm.datasets.sunspots.load_pandas().data
dta.index = pd.Index(sm.tsa.datetools.dates_from_range("1700", "2008"))
dta.index.freq = dta.index.inferred_freq
del dta["YEAR"]
dta.plot(figsize=(12, 8))

fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(dta.values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(dta, lags=40, ax=ax2)

# %%
# arma_mod20 = ARIMA(dta, order=(2, 0, 0)).fit()
# print(arma_mod20.params)

# arma_mod30 = ARIMA(dta, order=(3, 0, 0)).fit()
# print(arma_mod30.params)

# %% CO2 example
data_dict = {}
co2 = pd.read_csv(
    "data/co2_daily_mlo.csv",
    skiprows=32,
    names=["year", "month", "day", "timestamp", "ppm"],
)

ch4 = pd.read_csv(
    "data/ch4_mm_gl.csv",
    skiprows=46,
    names=["year", "month", "timestamp", "ppm", "std", "ppb_2", "std_2"],
)
ch4["ppm"] = ch4["ppm"] / 1000
X = jnp.array(co2["timestamp"])
Y = jnp.array(co2["ppm"]).astype(float)

p = np.polyfit(X, Y, 4)
print(p)
data = []
data = Y - np.polyval(p, X)
data = co2["ppm"].astype(float)

import statsmodels.api as sm


fig = plt.figure(figsize=(8, 10))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(data, lags=40, ax=ax1)
plt.xlabel("Timestep"), plt.ylabel("Autocorrelation")
plt.grid("on")
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(data, lags=40, ax=ax2)
plt.xlabel("Lags"), plt.ylabel("Autocorrelation Coefficient")
plt.grid("on")
plt.show()
plt.savefig("figures/CO2_autocorrelation_without_diff.png")

fig = plt.figure(figsize=(8, 10))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(np.diff(data), lags=40, ax=ax1)
plt.xlabel("Timestep"), plt.ylabel("Autocorrelation")
plt.grid("on")
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(np.diff(data), lags=40, ax=ax2)
plt.xlabel("Lags"), plt.ylabel("Autocorrelation Coefficient")
plt.grid("on")

plt.savefig("figures/CO2_autocorrelation_with_diff.png")
plt.show()

# %% do arima
data = np.array(co2["ppm"]).astype(float)
res = sm.tsa.arima.ARIMA(data[0 : len(data) // 2], order=(1, 1, 1)).fit()
print(res.summary())

plt.plot(data, label="Original")
plt.plot(res.fittedvalues[1:], label="Fitted", color="red")
plt.title("ARIMA Model Fit")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.show()

predictions = res.get_forecast(steps=500)

# Get the mean predicted values
mean_predictions = predictions.predicted_mean

# Get confidence intervals
conf_int = predictions.conf_int()

predict = res.get_prediction()
predict.predicted_mean.loc["1980-07-01":].plot(
    ax=ax, style="r--", label="One-step-ahead forecast"
)

res.plot_diagnostics(figsize=(12, 8))
# plt.show()
# # Plot confidence intervals
# plt.fill_between(
#     mean_predictions.index,
#     conf_int["lower y"],
#     conf_int["upper y"],
#     color="green",
#     alpha=0.2,
# )

plt.title("ARIMA Model Fit and Predictions")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.show()

# %%
fig, ax = plt.subplots()
plt.plot(X, Y)


model_output = res.predict()
plt.plot(X[1:], model_output[1:])
