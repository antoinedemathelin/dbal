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


def toy_example(size=1000, dim=2, cluster=10):
    Xs = 0.5 * np.random.randn(size, dim)
    mus = 2 * np.random.random((cluster, dim))
    sigmas = 0.5 * np.random.random((cluster,dim,dim))
    for i in range(len(sigmas)):
        sigmas[i, :, :] = 0.1 * sigmas[i, :, :].transpose().dot(sigmas[i, :, :])
        sigmas[i,:,:][np.diag_indices_from(sigmas[i,:,:])] *= 10

    mults = []
    for i in range(cluster):
        mults.append(np.random.multivariate_normal(mus[i], sigmas[i, :, :], size))
    mults = np.stack(mults, -1)
    espilon = np.random.choice(cluster, size)

    Xt = mults[np.arange(size), :, espilon]

    def f(X):
        return (1/2) * (X[:, 0] + 0.5 * X[:, 1]) + np.exp(X[:, 0] * X[:, 1] - X[:, 1]**2 - 0.25 * X[:, 0]**2)

    return Xs, Xt, f



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
    path = path + "digits/" + domain
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


def preprocessing_office(Xs, ys, Xt, yt):
    
    ohe = OneHotEncoder(sparse=False).fit(ys.reshape(-1, 1))
    ys_enc = ohe.transform(ys.reshape(-1, 1))
    yt_enc = ohe.transform(yt.reshape(-1, 1))
    
    def convert_y(y_pred):
        new_pred = np.zeros(y_pred.shape)
        args = np.argmax(y_pred, axis=1)
        new_pred[np.arange(len(new_pred)), args] = 1.
        return new_pred
    
    return Xs, ys_enc, Xt, yt_enc, convert_y


def preprocessing_digits(Xs, ys, Xt, yt):
    
    ohe = OneHotEncoder(sparse=False).fit(ys.reshape(-1, 1))
    ys_enc = ohe.transform(ys.reshape(-1, 1))
    yt_enc = ohe.transform(yt.reshape(-1, 1))
    
    def convert_y(y_pred):
        new_pred = np.zeros(y_pred.shape)
        args = np.argmax(y_pred, axis=1)
        new_pred[np.arange(len(new_pred)), args] = 1.
        return new_pred
    
    return Xs, ys_enc, Xt, yt_enc, convert_y


def preprocessing_superconductivity(Xs, ys, Xt, yt):
    maxes = Xs[:, 80:].max(0)
    Xs[:, 80:] = Xs[:, 80:] / (maxes + 1.e-8)
    Xt[:, 80:] = Xt[:, 80:] / (maxes + 1.e-8)
    std_sc = StandardScaler()
    std_sc.fit(Xs[:, :80])
    Xs[:, :80] = std_sc.transform(Xs[:, :80])
    Xt[:, :80] = std_sc.transform(Xt[:, :80])

    mu = np.mean(ys)
    std = np.std(ys)
    ys = (ys - mu) / std
    yt = (yt - mu) / std

    def convert_y(y_pred):
        return std * y_pred + mu
    
    return Xs, ys, Xt, yt, convert_y
