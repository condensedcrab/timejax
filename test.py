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

df = pd.read_csv("data/co2_daily_mlo.csv")
