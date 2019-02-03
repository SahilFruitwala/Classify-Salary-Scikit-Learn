# Classify-Salary-Scikit-Learn

In this Project using Logistic Regression Method I got 92% accuracy on both traing and tesing datasets.

You can find dataset from here. http://mlr.cs.umass.edu/ml/datasets/Adult 

I have done 2 things manually on datasets (On both training and testing datasets) :
1. Insert headers on both sets. (age,workclass,fnlwgt,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country,income)
2. Remove fullstop from testing set.

## Do Pre-Processing (preprocessing.py)
In this I have done these things:
1. Get data from csv file
2. Remove white spaces
3. Replace Global Constant '?' of missing values with numpy.NaN
4. Replace NaN with most_frequent data of respected columns
5. Replace Numpy array to again pandas dataframe
6. Return Dataframe

## Do One-Hot-Encoding (onehotencode.py)
In this I have done these things:
1.Here we have taken workclass, education, occupation, relationship, marital-status, race, sex, native-country as feature to do one-hot-encode
2. Decide of what feature we want one-hot-encoding and pass it to get_dummies() method
3. Drop feature column from dataframe
4. Join one-hot-encoded data to dataframe
5. return whole dataframe

## Do Training & Evaluation of Model (train.py)
In this I have done these things:
Start Training Data by Calling any methods and by giving traing data
1. Define features
2. Get Pre-Processed Train Data
3. Get One-Hot-Encoded Train Data
4. Define Input and Output Labels and Data of Training Set
5. Get Pre-Processed Test Data
6. Get One-Hot-Encoded Test Data
7. Define Input and Output Labels and Data of Testing Set
8. Apply Scaling to DATA
9. Call any Methods of TrainData Class which you want to use for training and evaluating model

### Availabe Meythods
1. Logistic Regression : do_logreg()
2. Decision Tree : do_clf()
3. K-Nearest Neighbors : do_knn()
4. Linear Discriminant Analysis : do_lda()
5. Gaussian Naive Bayes : do_gnb()
6. Support Vector Machine : do_svm()
