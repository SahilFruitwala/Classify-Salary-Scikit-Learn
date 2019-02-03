import pandas as pd
from preprocessing import PreProcessData

class OneHotEncode(object):
    """
    Create one-hot-encoding using pandas get_dummies() method.
    """
    def __init__(self,data):
        """ Get Data and initialized it """
        self.data = data
        # print(self.data.head(1))

    def do_one_hot_encode(self):
        """
        Here we have taken workclass, education, occupation, relationship, marital-status, race, sex, native-country as feature to do one-hot-encode
        Decide of what feature we want one-hot-encoding and pass it to get_dummies() method
        Drop feature column from dataframe
        Join one-hot-encoded data to dataframe
        return whole dataframe
        """
        # print(self.data.head(1))
        one_hot_encode = pd.get_dummies(self.data['workclass'])
        self.data = self.data.drop('workclass',axis=1)
        self.data = self.data.join(one_hot_encode)
        one_hot_encode = pd.get_dummies(self.data['education'])
        self.data = self.data.drop('education',axis=1)
        self.data = self.data.join(one_hot_encode)
        one_hot_encode = pd.get_dummies(self.data['occupation'])
        self.data = self.data.drop('occupation',axis=1)
        self.data = self.data.join(one_hot_encode)
        
        one_hot_encode = pd.get_dummies(self.data['marital-status'])
        self.data = self.data.drop('marital-status',axis=1)
        self.data = self.data.join(one_hot_encode)
        one_hot_encode = pd.get_dummies(self.data['relationship'])
        self.data = self.data.drop('relationship',axis=1)
        self.data = self.data.join(one_hot_encode)
        one_hot_encode = pd.get_dummies(self.data['race'])
        self.data = self.data.drop('race',axis=1)
        self.data = self.data.join(one_hot_encode)
        one_hot_encode = pd.get_dummies(self.data['sex'])
        self.data = self.data.drop('sex',axis=1)
        self.data = self.data.join(one_hot_encode)
        one_hot_encode = pd.get_dummies(self.data['native-country'])
        self.data = self.data.drop('native-country',axis=1)
        self.data = self.data.join(one_hot_encode)

        # print(self.data.head(1))
        return self.data
