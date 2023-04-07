import numpy as np
from rstl import STL

ts = np.arange(144)
freq = 12

stl = STL(ts, freq, "periodic")

trend = stl.trend

print(trend)