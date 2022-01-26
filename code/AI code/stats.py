import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from timeseries import read_data 

# data_2D.txt is in the data folder
input_file = 'data_2D.txt'

x1 = read_data(input_file, 2)
x2 = read_data(input_file, 3)

data = pd.DataFrame({'dim1': x1, 'dim2': x2})

print('\nMaximum values for each dimension:')
print(data.max())
print('\nMinimum values for each dimension:')
print(data.min())

print('\nOverall mean:')
print(data.mean())
print('\nRow-wise mean:')
print(data.mean(1)[:12])

data.rolling(center=False, window=24).mean().plot()
plt.title('Rolling mean')

print('\nCorrelation coefficients:\n', data.corr())

plt.figure()
plt.title('Rolling correlation')
data['dim1'].rolling(window=60).corr(other=data['dim2']).plot()

plt.show()
