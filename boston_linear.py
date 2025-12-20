import sys
sys.path.append('/home/aistudio/external-libraries')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
bos = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])

bos_target = raw_df.values[1::2, 2]
#boston=load_boston
#print(boston.key())
#print(boston.feature_names)

#bos = pd.DatatFrame(boston.data)
print(bos)
#bos_target = pd.DataFrame(boston.target)
print(bos_target)

X = bos[:, 5:6][:5]
Y = bos_target[:5]
plt.scatter(X, Y)
plt.xlabel(u'MEDV')
plt.ylabel(u'RM')
plt.title(u'The relation of RM and PRICE')
plt.show()

#X = np.array(X.values)
#Y = np.array(Y.values)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25)
X_train.shape,X_test.shape,Y_train.shape,Y_test.shape

lr=LinearRegression()
lr.fit(X_train,Y_train)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,)
print('求解截距项:', lr.intercept_)
print('求解系数:', lr.coef_)

Y_hat=lr.predict(X_test)
Y_hat[0:9]

plt.figure(figsize=(10,6))
t=np.arange(len(X_test))
plt.plot(t,Y_test,'r',linewidth=2,label='Y_test')
plt.plot(t,Y_hat,'g',linewidth=2,label='Y_hat')
plt.legend()
plt.xlabel('test data')
plt.ylabel('price')
plt.show()

plt.figure(figsize=(10,6))
plt.plot(Y_test,Y_hat,'o')
plt.plot([-10,60],[-10,60],color='red',linestyle='--',linewidth=1.5)
plt.plot([-10,60],[-10,60])
plt.xlabel('ground truth')
plt.ylabel('predicted')
plt.grid()
plt.show()

from sklearn import metrics
from sklearn.metrics import r2_score

print('r2:', lr.score(X_test,Y_test))
print('r2_score', r2_score(Y_test,Y_hat))
print('MAE:', metrics.mean_absolute_error(Y_test,Y_hat))
print('MSE:', metrics.mean_squared_error(Y_test,Y_hat))
print('RMSE:', np.sqrt(metrics.mean_squared_error(Y_test,Y_hat)))

