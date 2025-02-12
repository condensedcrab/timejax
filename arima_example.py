# %% ARIMA Example
# # following statsmodel example: https://www.statsmodels.org/stable/examples/notebooks/generated/tsa_arma_0.html#Exercise:-Can-you-obtain-a-better-fit-for-the-Sunspots-model?-(Hint:-sm.tsa.AR-has-a-method-select_order)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.api import qqplot

import jax
import jax.numpy as jnp

print(sm.datasets.sunspots.NOTE)

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
Y = jnp.array(co2["ppm"])


# %% write out predict (use Nx2 format)
@jax.jit
def predict(params: jnp.array, data: jnp.array):
    w, b = params
    return w * data + b
