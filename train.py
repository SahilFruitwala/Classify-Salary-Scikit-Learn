import pandas as pd
from preprocessing import PreProcessData
from onehotencode import OneHotEncode
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

class TrainData(object):
    def __init__(self):
        """
        Start Training Data by Calling any methods and by giving traing data
        Logistic Regression : do_logreg()
        Decision Tree : do_clf()
        K-Nearest Neighbors : do_knn()
        Linear Discriminant Analysis : do_lda()
        Gaussian Naive Bayes : do_gnb()
        Support Vector Machine : do_svm()
        """

    def do_logreg(self, X_train, y_train):
        """
        Logistic Regression : do_logreg(x,y)
        In x pass train data
        In y pass o/p label
        """
        model = LogisticRegression(solver='lbfgs',max_iter=1000,multi_class='auto')
        model.fit(X_train, y_train)
        print('Accuracy of Logistic regression classifier on training set: {:.2f}'.format(model.score(X_train, y_train)))
        return model

    def do_clf(self, X_train, y_train):
        """
        Decision Tree : do_clf(x,y)
        In x pass train data
        In y pass o/p label
        """
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        print('Accuracy of Decision Tree classifier on training set: {:.2f}'.format(model.score(X_train, y_train)))
        return model

    def do_knn(self, X_train, y_train):
        """
        K-Nearest Neighbors : do_knn(x,y)
        In x pass train data
        In y pass o/p label
        """
        model = KNeighborsClassifier()
        model.fit(X_train, y_train)
        print('Accuracy of K-NN classifier on training set: {:.2f}'.format(model.score(X_train, y_train)))
        return model

    def do_lda(self, X_train, y_train):
        """
        Linear Discriminant Analysis : do_lda(x,y)
        In x pass train data
        In y pass o/p label
        """
        model = LinearDiscriminantAnalysis()
        model.fit(X_train, y_train)
        print('Accuracy of LDA classifier on training set: {:.2f}'.format(model.score(X_train, y_train)))
        return model

    def do_gnb(self, X_train, y_train):
        """
        Gaussian Naive Bayes : do_gnb(x,y)
        In x pass train data
        In y pass o/p label
        """
        model = GaussianNB()
        model.fit(X_train, y_train)
        print('Accuracy of GNB classifier on training set: {:.2f}'.format(model.score(X_train, y_train)))
        return model

    def do_svm(self, X_train, y_train):
        """
        Support Vector Machine : do_svm(x,y)
        In x pass train data
        In y pass o/p label
        """
        model = SVC(gamma='auto')
        model.fit(X_train, y_train)
        print('Accuracy of SVM classifier on training set: {:.2f}'.format(model.score(X_train, y_train)))
        return model


if __name__ == "__main__":
    """
    Define features
    Get Pre-Processed Train Data
    Get One-Hot-Encoded Train Data
    Define Input and Output Labels and Data of Training Set
    Get Pre-Processed Test Data
    Get One-Hot-Encoded Test Data
    Define Input and Output Labels and Data of Testing Set

    Call any Methods of TrainData Class which you want to use for training and evaluating model
    """
    # feature_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'occupation', 'capital-gain', 'capital-loss', 'hours-per-week']
    # feature_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week']
    feature_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']

    train_data = PreProcessData('train.csv').start_preprocess()
    X_train = train_data[feature_names]
    X_train = OneHotEncode(X_train).do_one_hot_encode()
    y_train = train_data['income'].values

    test_data = PreProcessData('test.csv').start_preprocess()
    X_test = test_data[feature_names]
    X_test = OneHotEncode(X_test).do_one_hot_encode()
    y_test = test_data['income'].values

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = TrainData().do_logreg(X_train, y_train)
    print('Accuracy of Logistic regression classifier on test set: {:.2f}'.format(model.score(X_test, y_test)))
    # model = TrainData().do_clf(X_train, y_train)
    # print('Accuracy of Decision Tree classifier on test set: {:.2f}'.format(model.score(X_test, y_test)))
    # model = TrainData().do_knn(X_train, y_train)
    # print('Accuracy of K-NN classifier on test set: {:.2f}'.format(model.score(X_test, y_test)))
    # model = TrainData().do_lda(X_train, y_train)
    # print('Accuracy of LDA classifier on test set: {:.2f}'.format(model.score(X_test, y_test)))
    # model = TrainData().do_gnb(X_train, y_train)
    # print('Accuracy of GNB classifier on test set: {:.2f}'.format(model.score(X_test, y_test)))
    # model = TrainData().do_svm(X_train, y_train)
    # print('Accuracy of SVM classifier on test set: {:.2f}'.format(model.score(X_test, y_test)))    