# %%
# using CO2 and CH4 datasets from NOAA
# https://gml.noaa.gov/ccgg/trends/

# import common packages
import glob
import os
import matplotlib.pyplot as plt
import scipy
import numpy as np

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
    names=["year", "month", "timestamp", "ppb", "std", "ppb_2", "std_2"],
)

plt.figure()
plt.subplot(121)
plt.plot(co2)

# %%
