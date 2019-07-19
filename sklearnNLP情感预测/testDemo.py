
import utilHelpe

from sklearn import preprocessing
import numpy as np
X = np.array(['PDN0004178'])

print(X)


scaler= preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(X)
X_scaled = scaler.transform(X)

print('-----------',X_scaled)
X1=scaler.inverse_transform(X_scaled)
print('========',X1)
print('---------',X1[0, -1])
