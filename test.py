# %%
# using CO2 and CH4 datasets from NOAA
# https://gml.noaa.gov/ccgg/trends/

# import common packages
import glob
import os
import matplotlib.pyplot as plt
import scipy
import numpy as np
import seaborn as sns
import time
import pandas as pd
import jax.numpy as jnp

# %% load data and plot data

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

plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.plot(co2["timestamp"], co2["ppm"], color="maroon")
plt.xlabel("Time (yr)"), plt.ylabel("CO$_2$ Concentration (ppm)")
plt.grid("both")

plt.subplot(122)
plt.plot(ch4["timestamp"], ch4["ppm"], color="dodgerblue")
plt.xlabel("Time (yr)"), plt.ylabel("Avg. CH$_4$ Concentration (ppb)")
plt.grid("both")

# %% CO2 data - 0th order FFT analysis
X = co2["timestamp"].to_numpy()
Y = co2["ppm"].to_numpy()

timestep = jnp.diff(X)[0]
N = len(X)
desired_N = 2**16


ft_input = np.pad(Y,mode='constant')

freq = np.fft.fftfreq(N, timestep)[: N // 2]
F = jnp.fft.fft(input)

plt.figure()
plt.plot(freq[:], np.abs(F[ : N // 2]))
plt.xlim([0,10])
plt.yscale("log")
plt.xlabel("Frequency (yr$^{-1}$)"), plt.ylabel("FT Magnitude (arb. units)")

# %% sliding window analysis
window_size = 4
time = co2["timestamp"]

for i in range()