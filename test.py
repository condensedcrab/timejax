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

# %% load data

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
plt.plot(co2["timestamp"], co2["ppm"])
plt.xlabel("Time (yr)"), plt.ylabel("CO$_2$ Concentration (ppm)")
plt.grid("both")

plt.subplot(122)
plt.plot(ch4["timestamp"], ch4["ppm"])
plt.xlabel("Time (yr)"), plt.ylabel("Avg. CH$_4$ Concentration (ppb)")
plt.grid("both")

# %%
