import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer


class PreProcessData(object):
    """
    Get data from csv file
    Remove white spaces
    Replace Global Constant '?' of missing values with numpy.NaN
    Replace NaN with most_frequent data of respected columns
    """

    def __init__(self, file_name):
        """ Get CSV File Name"""
        super(PreProcessData, self).__init__()
        self.file_name = file_name

    def start_preprocess(self):
        """ 
        Remove white spaces
        Replace Global Constant '?' of missing values with numpy.NaN
        Replace NaN with most_frequent data of respected columns
        Replace Numpy array to again pandas dataframe
        """
        data = pd.read_csv(self.file_name)
        # data = pd.read_csv(self.file_name,usecols=[0,1,2,3,4,6,10,11,12,14])
        # data = pd.read_csv(self.file_name,usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14])
        data = data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        # print((data[['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income']] == '?').sum())

        # data[['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'occupation', 'capital-gain', 'capital-loss', 'hours-per-week', 'income']] = data[['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'occupation', 'capital-gain', 'capital-loss', 'hours-per-week', 'income']].replace('?', np.NaN)
        data[['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']] = data[['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']].replace('?', np.NaN)
        # data[['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'income']] = data[['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'income']].replace('?', np.NaN)
        # data = data.replace('?',np.NaN)

        x = data.iloc[:, :].values
        # print(data.head(30))
        imputer = SimpleImputer(strategy="most_frequent", verbose=1)
        imputer = imputer.fit(x[:, :])
        x[:, :] = imputer.transform(x[:, :])

        # final_data = pd.DataFrame({'age':x[:,0],'workclass':x[:,1],'fnlwgt':x[:,2],'education':x[:,3],'education-num':x[:,4],'occupation':x[:,5],'capital-gain':x[:,6],'capital-loss':x[:,7],'hours-per-week':x[:,8],'income':x[:,9]})
        final_data = pd.DataFrame({'age':x[:,0],'workclass':x[:,1],'fnlwgt':x[:,2],'education':x[:,3],'education-num':x[:,4],'marital-status':x[:,5],'occupation':x[:,6],'relationship':x[:,7],'race':x[:,8],'sex':x[:,9],'capital-gain':x[:,10],'capital-loss':x[:,11],'hours-per-week':x[:,12],'income':x[:,13],'native-country':x[:,14]})
        # final_data = pd.DataFrame({'age':x[:,0], 'workclass':x[:,1], 'fnlwgt':x[:,2], 'education':x[:,3], 'education-num':x[:,4], 'marital-status':x[:,5], 'occupation':x[:,6], 'relationship':x[:,7], 'race':x[:,8], 'sex':x[:,9], 'capital-gain':x[:,10], 'capital-loss':x[:,11], 'hours-per-week':x[:,12], 'native-country':x[:,13], 'income':x[:,14]})
        # print(final_data.head(40))
        return final_data
