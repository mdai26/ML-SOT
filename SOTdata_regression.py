# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 10:50:18 2019

@author: Minyi Dai
"""

print(__doc__)


# Code source: Jaques Grobler
# License: BSD 3 clause


import matplotlib.pyplot as plt
import numpy as np
import csv
import sklearn.neighbors as sn
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets.base import Bunch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern,RationalQuadratic
from sklearn import tree
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

def load_machine_data():
    with open('SOTdata_regression.csv') as csv_file:
        data_file = csv.reader(csv_file)
        temp=next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data=np.empty((n_samples,n_features))
        target=np.empty((n_samples,),dtype=np.float64)
        
        for i, sample in enumerate(data_file):
            data[i] = np.asarray(sample[:-1],dtype=np.float64)
            target[i] = np.asarray(sample[-1],dtype=np.float64)
            
    return Bunch(data=data,target=target)

# Load the diabetes dataset
mydata=load_machine_data()

# Use only one feature
mydata_X = mydata.data

# Split the data into training/testing sets
mydata_X_train = mydata_X[:-80]
mydata_X_test = mydata_X[-80:]

# Split the targets into training/testing sets
mydata_Y_train = mydata.target[:-80]
mydata_Y_test = mydata.target[-80:]

# Create linear regression object
regr1 = linear_model.LinearRegression()
regr2 = linear_model.Lasso(alpha=0.1)
regr3 = linear_model.BayesianRidge()
regr4 = linear_model.Ridge(alpha=0.1)
regr5 = sn.KNeighborsRegressor(n_neighbors=10)
regr6_kernel = Matern(length_scale = 1.0,nu=0.05)+RationalQuadratic(length_scale=1.0,alpha=1.0)
regr6 = GaussianProcessRegressor(kernel=regr6_kernel)
regr7 = tree.DecisionTreeRegressor(max_depth=30)

# Train the model using the training sets
regr1.fit(mydata_X_train, mydata_Y_train)
regr2.fit(mydata_X_train, mydata_Y_train)
regr3.fit(mydata_X_train, mydata_Y_train)
regr4.fit(mydata_X_train, mydata_Y_train)
regr5.fit(mydata_X_train, mydata_Y_train)
regr6.fit(mydata_X_train, mydata_Y_train)
regr7.fit(mydata_X_train, mydata_Y_train)

# Make predictions using the testing set
mydata_Y1_pred = regr1.predict(mydata_X_test)
mydata_Y2_pred = regr2.predict(mydata_X_test)
mydata_Y3_pred = regr3.predict(mydata_X_test)
mydata_Y4_pred = regr4.predict(mydata_X_test)
mydata_Y5_pred = regr5.predict(mydata_X_test)
mydata_Y6_pred = regr6.predict(mydata_X_test)
mydata_Y7_pred = regr7.predict(mydata_X_test)

# The coefficients
print('Coefficients of 1: \n', regr1.coef_)
print("Mean squared error of 1: %.2f"
      % mean_squared_error(mydata_Y_test,mydata_Y1_pred))
print('Variance score of 1: %.2f' % r2_score(mydata_Y_test, mydata_Y1_pred))

print('Coefficients of 2: \n', regr2.coef_)
print("Mean squared error of 2: %.2f"
      % mean_squared_error(mydata_Y_test,mydata_Y2_pred))
print('Variance score of 2: %.2f' % r2_score(mydata_Y_test, mydata_Y2_pred))

print('Coefficients of 3: \n', regr3.coef_)
print("Mean squared error of 3: %.2f"
      % mean_squared_error(mydata_Y_test,mydata_Y3_pred))
print('Variance score of 3: %.2f' % r2_score(mydata_Y_test, mydata_Y3_pred))

print('Coefficients of 4: \n', regr4.coef_)
print("Mean squared error of 4: %.2f"
      % mean_squared_error(mydata_Y_test,mydata_Y4_pred))
print('Variance score of 4: %.2f' % r2_score(mydata_Y_test, mydata_Y4_pred))

print("Mean squared error of 5: %.2f"
      % mean_squared_error(mydata_Y_test,mydata_Y5_pred))
print('Variance score of 5: %.2f' % r2_score(mydata_Y_test, mydata_Y5_pred))

print("Mean squared error of 6: %.2f"
      % mean_squared_error(mydata_Y_test,mydata_Y6_pred))
print('Variance score of 6: %.2f' % r2_score(mydata_Y_test, mydata_Y6_pred))

print("Mean squared error of 7: %.2f"
      % mean_squared_error(mydata_Y_test,mydata_Y7_pred))
print('Variance score of 7: %.2f' % r2_score(mydata_Y_test, mydata_Y7_pred))

a = [-1,0.1,1]
b=a
plt.figure(0)
# Plot outputs
#plt.scatter(mydata_Y_test, mydata_Y1_pred, color='blue')
#plt.scatter(mydata_Y_test, mydata_Y2_pred, color='red')
#plt.scatter(mydata_Y_test, mydata_Y3_pred, color='yellow')
#plt.scatter(mydata_Y_test, mydata_Y4_pred, color='green')
#plt.scatter(mydata_Y_test, mydata_Y5_pred, color='red')
#plt.scatter(mydata_Y_test, mydata_Y6_pred, color='yellow')
plt.scatter(mydata_Y_test, mydata_Y7_pred, color='green')
plt.plot(a,b,color='black',linewidth=3)

plt.xlim((-1,1))
plt.ylim((-1,1))

plt.show()
#
#file=open("prediction_model7.dat","w")
#for i in range(0,80):
#    file.write("%6d%8.2f%8.2f\r"
#               %(i,mydata_Y_test[i],mydata_Y7_pred[i]))
#file.close

colors=["green","yellow","red","blue","cyan","magenta"]
plt.figure(1)
for count, degree in enumerate([5,6,7,8]):
    model = make_pipeline(PolynomialFeatures(degree),Ridge())
    model.fit(mydata_X_train, mydata_Y_train)
    Y_predict = model.predict(mydata_X_test)
    print("Mean squared error of degree %d: %.2f"
      %(degree,mean_squared_error(mydata_Y_test,Y_predict)))
    print('Variance score of degree %d: %.2f' %(degree,r2_score(mydata_Y_test, Y_predict)))
    plt.scatter(mydata_Y_test,Y_predict,color=colors[count],label="degree %d" %degree)
#    for i in range(0,20):
#        print("%6d%8.2f%8.2f\r" %(i,mydata_Y_test[i],Y_predict[i]))
    
    
plt.plot(a,b,color='black',linewidth=3)
plt.xlim((-1,1))
plt.ylim((-1,1))

plt.show()


