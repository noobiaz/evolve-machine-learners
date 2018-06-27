import numpy as np
from scipy import stats

import matplotlib
import matplotlib.pyplot as plt
import scipy.stats

matplotlib.style.use('ggplot')

np.random.seed(1)
data = np.round(np.random.normal(5, 2, 100))
print("data:")
print(data)

mean=np.mean(data)
print("mean:")
print(mean)

median=np.median(data)
print("median:")
print(median)

mode=stats.mode(data)
print("mode:")
print(mode)

range=np.ptp(data)
print("range:")
print(range)

var=np.var(data)
print("var:")
print(var)

std=np.std(data)
print("std:")
print(std)

plt.hist(data, bins=10, range=(0,10), edgecolor='black')
plt.show()