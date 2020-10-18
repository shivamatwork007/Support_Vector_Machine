import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics

cancer = datasets.load_breast_cancer()

#print(cancer.target_names)
#print(cancer.feature_names)
X = cancer.data
Y = cancer.target

X_train,X_test,Y_train,Y_test = sklearn.model_selection.train_test_split(X,Y, test_size = 0.1)

classes = ["malignant","benign"]

clf = svm.SVC(kernel= "linear",C=1)
clf.fit(X_train,Y_train)

predicted = clf.predict(X_test)

accuracy = metrics.accuracy_score(Y_test,predicted)
print(accuracy)