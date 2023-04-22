from pathlib import Path
from django.conf import settings
import matplotlib.pyplot as plt
import numpy as np
# %matplotlib notebook
from sklearn import svm, metrics, datasets
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.linear_model import Perceptron
from sklearn.metrics import mean_squared_error
from math import sqrt
from skimage.io import imread
from skimage.transform import resize


def load_image_files(container_path, dimension=(64, 64)):

    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    categories = [fo.name for fo in folders]

    descr = "A image classification dataset"
    images = []
    flat_data = []
    target = []
    for i, direc in enumerate(folders):
        for file in direc.iterdir():
            img = imread(file)
            img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')
            flat_data.append(img_resized.flatten())
            images.append(img_resized)
            target.append(i)
    flat_data = np.array(flat_data)
    target = np.array(target)
    images = np.array(images)

    return Bunch(data=flat_data,
                 target=target,
                 target_names=categories,
                 images=images,
                 DESCR=descr)

class UserSVMCode:
    def startSvm(self):
        #filepath = settings.MEDIA_ROOT + "\\datasets\\"
        filepath = settings.MEDIA_ROOT
        image_dataset = load_image_files(filepath)

        X_train, X_test, y_train, y_test = train_test_split(
            image_dataset.data, image_dataset.target, test_size=0.3, random_state=109)

        param_grid = [
            {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
            {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
        ]
        svc = svm.SVC()
        clf = GridSearchCV(svc, param_grid)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        #print("Classification report for - \n{}:\n{}\n".format(clf, metrics.classification_report(y_test, y_pred)))
        print("SVM Classification report for - :\n{}\n".format(metrics.classification_report(y_test, y_pred)))
        meanRmaseSVM = sqrt(mean_squared_error(y_test, y_pred))
        print("SVm RSME= ",meanRmaseSVM)
        return meanRmaseSVM

    def startKnn(self):
        filepath = settings.MEDIA_ROOT
        image_dataset = load_image_files(filepath)
        X_train, X_test, y_train, y_test = train_test_split(image_dataset.data, image_dataset.target, test_size=0.3, random_state=109)
        neigh = KNeighborsClassifier(n_neighbors=3)
        neigh.fit(X_train, y_train)
        y_pred = neigh.predict(X_test)
        print("KNN Classification report for - :\n{}\n".format(metrics.classification_report(y_test, y_pred)))
        meanRmaseSVM = sqrt(mean_squared_error(y_test, y_pred))
        print("KNN Mean RSME= ", meanRmaseSVM)
        return meanRmaseSVM

    def startDecisionTree(self):
        filepath = settings.MEDIA_ROOT
        image_dataset = load_image_files(filepath)
        X_train, X_test, y_train, y_test = train_test_split(image_dataset.data, image_dataset.target, test_size=0.3, random_state=109)
        dt = tree.DecisionTreeClassifier()
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_test)
        print("DT Classification report for - :\n{}\n".format(metrics.classification_report(y_test, y_pred)))
        meanRmaseSVM = sqrt(mean_squared_error(y_test, y_pred))
        print("DT Mean RSME= ", meanRmaseSVM)
        return meanRmaseSVM

    def startSLP(self):
        filepath = settings.MEDIA_ROOT
        image_dataset = load_image_files(filepath)
        X_train, X_test, y_train, y_test = train_test_split(image_dataset.data, image_dataset.target, test_size=0.3, random_state=109)
        clf = Perceptron(tol=1e-3, random_state=0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print("SLP Classification report for - :\n{}\n".format(metrics.classification_report(y_test, y_pred)))
        meanRmaseSVM = sqrt(mean_squared_error(y_test, y_pred))
        print("SLP Mean RSME= ", meanRmaseSVM)
        return meanRmaseSVM







