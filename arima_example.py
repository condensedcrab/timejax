# %% ARIMA Example
# # following statsmodel example: https://www.statsmodels.org/stable/examples/notebooks/generated/tsa_arma_0.html#Exercise:-Can-you-obtain-a-better-fit-for-the-Sunspots-model?-(Hint:-sm.tsa.AR-has-a-method-select_order)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.api import qqplot

print(sm.datasets.sunspots.NOTE)
