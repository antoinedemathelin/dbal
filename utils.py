"""
Utility functions
"""

import os
import sys
import urllib
import zipfile
import gzip
import tarfile
import shutil

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PATH = "../datasets/"


def download_superconductivity(path=PATH):
    urllib.request.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/00464/superconduct.zip",
                               path + "superconductivity.zip")
    
    with zipfile.ZipFile(path + "superconductivity.zip", 'r') as zip_ref:
        zip_ref.extractall(path + "superconductivity")
    os.remove(path + "superconductivity.zip")


def load_office(domain, path=PATH):
    path = path + "office_preprocessed/" + domain
    X = np.load(path+'_X.npy')
    y = np.load(path+'_y.npy')
    return X, y


def load_digits(domain, path=PATH):
    path = path + "digits_preprocessed/" + domain
    X = np.load(path+'_X.npy')
    y = np.load(path+'_y.npy')
    return X, y


def load_superconductivity(domain=None, path=PATH):
    """
    Load superconductivity dataset
    """
    path = path + "superconductivity/"
    data = pd.read_csv(path + "train.csv")
    
    formula = pd.read_csv(path + "unique_m.csv")
    formula = formula.drop(["material", "critical_temp"], 1)
    
    split_col = (data.corr().iloc[:, -1].abs()
                 - 0.3).abs().sort_values().head(1).index[0]
    cuts = np.percentile(data[split_col].values, [25, 50, 75])
    
    X = data.drop([data.columns[-1], split_col], 1).__array__()
    X = np.concatenate((X, formula.values), axis=1)
    
    y = data[data.columns[-1]].__array__()
    
    keep_index = ~pd.DataFrame(X).duplicated().values
    X = X[keep_index]
    y = y[keep_index]
    
    split_col = list(data.columns).index(split_col)
    
    if domain is None:
        return X, y
    else:
        index = _superconductivity_domain(data.values[keep_index], cuts, split_col, int(domain))
        return X[index], y[index]


def _superconductivity_domain(data, cuts, split_col, i):
    """
    Get indexes of ith split of data
    """
    if i == 0:
        return np.argwhere(data[:, split_col] <= cuts[0]).ravel()
    elif i == len(cuts) or i == -1:
        return np.argwhere(data[:, split_col] > cuts[-1]).ravel()
    else:
        return np.argwhere((data[:, split_col] <= cuts[i])
                           & (data[:, split_col] > cuts[i-1])).ravel()


def preprocessing_office(X, y):

    y_enc = OneHotEncoder(sparse=False).fit_transform(y.reshape(-1, 1))
    
    def convert_y(y_pred):
        new_pred = np.zeros(y_pred.shape)
        args = np.argmax(y_pred, axis=1)
        new_pred[np.arange(len(new_pred)), args] = 1.
        return new_pred
    
    return X, y_enc, convert_y


def preprocessing_digits(X, y):
    
    y_enc = OneHotEncoder(sparse=False).fit_transform(y.reshape(-1, 1))
    
    def convert_y(y_pred):
        new_pred = np.zeros(y_pred.shape)
        args = np.argmax(y_pred, axis=1)
        new_pred[np.arange(len(new_pred)), args] = 1.
        return new_pred
    
    return X, y_enc, convert_y


def preprocessing_superconductivity(X, y, src_index):
    X[:, 80:] = X[:, 80:] / (X[src_index, 80:].max(0) + 1.e-8)
    std_sc = StandardScaler()
    std_sc.fit(X[src_index, :80])
    X[:, :80] = std_sc.transform(X[:, :80])

    y_log = np.log(y + 1)
    mu = np.mean(y_log[src_index])
    std = np.std(y_log[src_index])
    y = (y_log - mu) / std

    def convert_y(y_pred):
        return np.exp(std * y_pred + mu) - 1
    
    return X, y, convert_y
