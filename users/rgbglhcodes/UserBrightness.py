from sklearn import svm, metrics, datasets
from sklearn.utils import Bunch
from sklearn.model_selection import  train_test_split
import matplotlib.pyplot as plt
from matplotlib import style
#style.use("ggplot")
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.linear_model import Perceptron
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import numpy as np
class UserImageBrightness:
    def startSvm(self,df):
        X = df[['redColor', 'greenColor', 'blueColor']]
        y = df['picbrightness']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,  random_state=109)
        print(df.head())
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        meanRmaseSVM = sqrt(mean_squared_error(y_test, y_pred))
        print("SVM RSME= ", meanRmaseSVM)
        return meanRmaseSVM

    def startKnn(self,df):
        X = df[['redColor', 'greenColor', 'blueColor']]
        y = df['picbrightness']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=109)
        print(df.head())
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        clf = KNeighborsClassifier(n_neighbors=3)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print("KNN Classification report for - :\n{}\n".format(metrics.classification_report(y_test, y_pred)))
        meanRmaseSVM = sqrt(mean_squared_error(y_test, y_pred))
        print("KNN Mean RSME= ", meanRmaseSVM)
        return meanRmaseSVM

    def startDecisionTree(self,df):
        X = df[['redColor', 'greenColor', 'blueColor']]
        y = df['picbrightness']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=109)
        print(df.head())
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        clf = tree.DecisionTreeClassifier()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print("DT Classification report for - :\n{}\n".format(metrics.classification_report(y_test, y_pred)))
        meanRmaseSVM = sqrt(mean_squared_error(y_test, y_pred))
        print("DT Mean RSME= ", meanRmaseSVM)
        return meanRmaseSVM

    def startSLP(self,df):
        X = df[['redColor', 'greenColor', 'blueColor']]
        y = df['picbrightness']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=109)
        print(df.head())
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        clf = Perceptron(tol=1e-3, random_state=0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print("SLP Classification report for - :\n{}\n".format(metrics.classification_report(y_test, y_pred)))
        meanRmaseSVM = sqrt(mean_squared_error(y_test, y_pred))
        print("SLP Mean RSME= ", meanRmaseSVM)
        return meanRmaseSVM
