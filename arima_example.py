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


fig = plt.figure(figsize=(12, 12))
ax1 = fig.add_subplot(221)
fig = sm.graphics.tsa.plot_acf(data, lags=40, ax=ax1)
plt.xlabel("Lags"), plt.ylabel("Autocorrelation")
plt.title("Autocorrelation (No differencing)")

plt.grid("on")
ax2 = fig.add_subplot(222)
fig = sm.graphics.tsa.plot_pacf(data, lags=40, ax=ax2)
plt.xlabel("Lags"), plt.ylabel("Autocorrelation Coefficient")
plt.grid("on")

ax1 = fig.add_subplot(223)
fig = sm.graphics.tsa.plot_acf(np.diff(data), lags=40, ax=ax1)
plt.xlabel("Lags"), plt.ylabel("Autocorrelation")
plt.title("Autocorrelation (Diff.)")
plt.grid("on")
ax2 = fig.add_subplot(224)
fig = sm.graphics.tsa.plot_pacf(np.diff(data), lags=40, ax=ax2)
plt.xlabel("Lags"), plt.ylabel("Autocorrelation Coefficient")
plt.grid("on")
plt.title("Partial Autocorrelation (Diff.)")

plt.savefig("figures/CO2_autocorrelation.png")
plt.show()

# %% do arima

res = sm.tsa.arima.ARIMA(data[0 : len(data)], order=(1, 1, 1)).fit()
print(res.summary())

fig = plt.figure(figsize=(12, 4))
ax = plt.subplot(121)
plt.plot(X[1:], res.resid[1:])
plt.xlabel("Time"), plt.ylabel("Model Residual (ppm)")
plt.grid("on")

ax = plt.subplot(122)
plt.hist(res.resid[1:], bins=np.linspace(-4, 4, 25))
plt.xlabel("Residual (ppm)"), plt.ylabel("Counts")
plt.grid("on")

plt.savefig("figures/ARIMA_111_CO2.png")
plt.show()


# %% SARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX

data = co2[["timestamp", "ppm"]]


result = seasonal_decompose(data, model="additive")
fig = result.plot()
# %%
