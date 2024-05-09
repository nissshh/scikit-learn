from sklearn import datasets
from sklearn import svm
import numpy as np

######Sample Data Set#######
iris_data = datasets.load_iris()
digits = datasets.load_digits()
#print(iris_data)
# print(iris_data.keys)
clf=svm.SVC(gamma=0.001,C=100.)

######Fit learning from Data Set#######
#####data is input context data 
##### target is outcome of the context

clf.fit(iris_data.data[:],iris_data.target[:])
print(clf)
# pred = clf.predict(iris_data.data[-5:])

######- Test Data Set#######
arr = np.array([[6.5, 3. , 5.5, 1.8],
        [7.7, 3.8, 6.7, 2.2],
       [7.7, 2.6, 6.9, 2.3],
       [6. , 2.2, 5. , 1.5],
       [6.9, 3.2, 5.7, 2.3],
       [5.6, 2.8, 4.9, 2. ],
       [7.7, 2.8, 6.7, 2. ],
       [6.3, 2.7, 4.9, 1.8],
       [4.8, 3.4, 1.6, 0.2],
       [4.8, 3. , 1.4, 0.1],
       [4.3, 3. , 1.1, 0.1],
       [5.8, 4. , 1.2, 0.2],
       [5.7, 4.4, 1.5, 0.4],
       [5.4, 3.9, 1.3, 0.4],
       [5.1, 3.5, 1.4, 0.3],
       [5.7, 3.8, 1.7, 0.3],
       [5.1, 3.8, 1.5, 0.3]])

###### prediction on test data set
pred = clf.predict(arr[:])
print(type(pred))
print(pred)
print(pred.any)
print(type(iris_data.target_names))
print(iris_data.target_names.dtype)
print(iris_data.target_names)
loc = pred.max()
print(loc)
loc = pred.flat
for f in loc:
    print(iris_data.target_names[f])
# print(iris_data.target_names[loc])
