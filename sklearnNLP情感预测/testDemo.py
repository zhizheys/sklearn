import numpy as np

x = np.array([[1,2,3,4],
  [2,3,np.nan,5],
  [np.nan,5,2,3]])
print(np.argwhere(np.isnan(x)))