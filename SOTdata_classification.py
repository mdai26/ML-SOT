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
from sklearn.datasets.base import Bunch
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn import svm
from sklearn.svm import NuSVC
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier

def load_machine_data():
    with open('SOTdata_classification.csv') as csv_file:
        data_file = csv.reader(csv_file)
        temp=next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        data=np.empty((n_samples,n_features))
        target=np.empty((n_samples,),dtype=np.int64)
        
        for i, sample in enumerate(data_file):
            data[i] = np.asarray(sample[:-1],dtype=np.float64)
            target[i] = np.asarray(sample[-1],dtype=np.int64)
            
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
clas1=LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial',max_iter=1000)
clas2=SGDClassifier(max_iter=1000,tol=1e-3)
clas3=Perceptron(tol=1e-3,random_state=0)
clas4=PassiveAggressiveClassifier(max_iter=1000,random_state=0)
clas5=svm.SVC(gamma='auto')
clas6=NuSVC(gamma='auto')
clas7=LinearSVC(random_state=0,tol=1e-5)
clas8=KNeighborsClassifier(n_neighbors=5)
clas9=GaussianProcessClassifier(random_state=0)
clas10=DecisionTreeClassifier(random_state=0,max_depth=14)

# Train the model using the training sets
clas1.fit(mydata_X_train, mydata_Y_train)
clas2.fit(mydata_X_train, mydata_Y_train)
clas3.fit(mydata_X_train, mydata_Y_train)
clas4.fit(mydata_X_train, mydata_Y_train)
clas5.fit(mydata_X_train, mydata_Y_train)
clas6.fit(mydata_X_train, mydata_Y_train)
clas7.fit(mydata_X_train, mydata_Y_train)
clas8.fit(mydata_X_train, mydata_Y_train)
clas9.fit(mydata_X_train, mydata_Y_train)
clas10.fit(mydata_X_train, mydata_Y_train)

# Make predictions using the testing set
mydata_Y1_pred = clas1.predict(mydata_X_test)
mydata_Y2_pred = clas2.predict(mydata_X_test)
mydata_Y3_pred = clas3.predict(mydata_X_test)
mydata_Y4_pred = clas4.predict(mydata_X_test)
mydata_Y5_pred = clas5.predict(mydata_X_test)
mydata_Y6_pred = clas6.predict(mydata_X_test)
mydata_Y7_pred = clas7.predict(mydata_X_test)
mydata_Y8_pred = clas8.predict(mydata_X_test)
mydata_Y9_pred = clas9.predict(mydata_X_test)
mydata_Y10_pred = clas10.predict(mydata_X_test)


# The coefficients
print('Variance score of 1: %.2f' % accuracy_score(mydata_Y_test, mydata_Y1_pred))
print('Variance score of 2: %.2f' % accuracy_score(mydata_Y_test, mydata_Y2_pred))
print('Variance score of 3: %.2f' % accuracy_score(mydata_Y_test, mydata_Y3_pred))
print('Variance score of 4: %.2f' % accuracy_score(mydata_Y_test, mydata_Y4_pred))
print('Variance score of 5: %.2f' % accuracy_score(mydata_Y_test, mydata_Y5_pred))
print('Variance score of 6: %.2f' % accuracy_score(mydata_Y_test, mydata_Y6_pred))
print('Variance score of 7: %.2f' % accuracy_score(mydata_Y_test, mydata_Y7_pred))
print('Variance score of 8: %.2f' % accuracy_score(mydata_Y_test, mydata_Y8_pred))
print('Variance score of 9: %.2f' % accuracy_score(mydata_Y_test, mydata_Y9_pred))
print('Variance score of 10: %.2f' % accuracy_score(mydata_Y_test, mydata_Y10_pred))

a = [0,0.5,4]
b=a
plt.figure(0)
# Plot outputs
plt.scatter(mydata_Y_test, mydata_Y1_pred, color='blue')
plt.scatter(mydata_Y_test, mydata_Y2_pred, color='red')
plt.scatter(mydata_Y_test, mydata_Y3_pred, color='yellow')
plt.scatter(mydata_Y_test, mydata_Y4_pred, color='green')
plt.scatter(mydata_Y_test, mydata_Y5_pred, color='red')
plt.scatter(mydata_Y_test, mydata_Y6_pred, color='yellow')
plt.scatter(mydata_Y_test, mydata_Y7_pred, color='green')
plt.scatter(mydata_Y_test, mydata_Y8_pred, color='yellow')
plt.scatter(mydata_Y_test, mydata_Y9_pred, color='green')
plt.scatter(mydata_Y_test, mydata_Y10_pred, color='yellow')

plt.plot(a,b,color='black',linewidth=3)

plt.xlim((0,4))
plt.ylim((0,4))

plt.show()


#file=open("classification_prediction_model10.dat","w")
#for i in range(0,80):
#    file.write("%6d%6d%6d\r"
#               %(i,mydata_Y_test[i],mydata_Y10_pred[i]))
#file.close



