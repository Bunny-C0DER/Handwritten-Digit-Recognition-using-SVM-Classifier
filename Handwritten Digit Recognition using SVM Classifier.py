#importing libraries

import numpy as np
from sklearn.datasets import load_digits

#loading dataset

dataset = load_digits()

#summarizing dataset

print(dataset.data)
print(dataset.target)
print(dataset.data.shape)
print(dataset.images.shape)

dataimageLength = len(dataset.images)
print(dataimageLength)

#visualizing dataset

n=9  #Number of the sample out of 1797 samples
import matplotlib.pyplot as plt
plt.gray()
plt.matshow(dataset.images[n])
plt.show()

dataset.images[n]

#segregating dataset into X(Input/Independent Variable) and Y(Output/Dependent Variable)
X = dataset.images.reshape((dataimageLength,-1))
X

Y = dataset.target
Y

#splitting dataset into Train & Test

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state = 0)
print(X_train.shape)
print(X_test.shape)

#training

from sklearn import svm
model = svm.SVC(kernel='linear')
model.fit(X_train,Y_train)

#predicting, what the digit is from test data

n=14
result = model.predict(dataset.images[n].reshape((1,-1)))
plt.imshow(dataset.images[n], cmap=plt.cm.gray_r, interpolation='nearest')
print(result)
plt.axis('off')
plt.title('%i' %result)
plt.show()

#prediction for test data

Y_pred = model.predict(X_test)
print(np.concatenate((Y_pred.reshape(len(Y_pred),1), Y_test.reshape(len(Y_test),1)),1))

#prediction for test data

Y_pred = model.predict(X_test)
print(np.concatenate((Y_pred.reshape(len(Y_pred),1), Y_test.reshape(len(Y_test),1)),1))



#----------------------------------------------------------------------


#playing with different method

from sklearn import svm
model1 = svm.SVC(kernel='linear')
model2 = svm.SVC(kernel='rbf')
model3 = svm.SVC(gamma=0.001)
model4 = svm.SVC(gamma=0.0001,C=0.1)
model5 = svm.SVC(gamma=0.001,C=0.2)
model6 = svm.SVC(gamma=0.002,C=0.1)

model1.fit(X_train,Y_train)
model2.fit(X_train,Y_train)
model3.fit(X_train,Y_train)
model4.fit(X_train,Y_train)
model5.fit(X_train,Y_train)
model6.fit(X_train,Y_train)

Y_predModel1 = model1.predict(X_test)
Y_predModel2 = model2.predict(X_test)
Y_predModel3 = model3.predict(X_test)
Y_predModel4 = model4.predict(X_test)
Y_predModel5 = model5.predict(X_test)
Y_predModel6 = model6.predict(X_test)

print("Accuracy of the Model 1: {0}%".format(accuracy_score(Y_test, Y_predModel1)*100))
print("Accuracy of the Model 2: {0}%".format(accuracy_score(Y_test, Y_predModel2)*100))
print("Accuracy of the Model 3: {0}%".format(accuracy_score(Y_test, Y_predModel3)*100))
print("Accuracy of the Model 4: {0}%".format(accuracy_score(Y_test, Y_predModel4)*100))
print("Accuracy of the Model 5: {0}%".format(accuracy_score(Y_test, Y_predModel5)*100))
print("Accuracy of the Model 6: {0}%".format(accuracy_score(Y_test, Y_predModel6)*100))