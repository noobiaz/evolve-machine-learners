import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(1)
matplotlib.style.use('ggplot')

x = np.random.randint(0, 50, 1000)
y = np.random.randint(0, 50, 1000)

corrcoeffs=stats.spearmanr(x, y)
print("corrcoeffs:")
print(corrcoeffs)

plt.scatter(x, y)
plt.show()