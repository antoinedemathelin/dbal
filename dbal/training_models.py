"""
Training models
"""

import inspect

import numpy as np


class TrainingModel:
    
    def __init__(self, get_model, **kwargs):
        self.get_model = get_model
        self.kwargs = kwargs
        
    def fit(self, Xs, ys, Xt=None, yt=None, **fit_params):
        
        if "input_shape" in inspect.signature(self.get_model).parameters:
            self.model = self.get_model(input_shape=Xs.shape[1:],
                                        output_shape=ys.shape[1:],
                                        **self.kwargs)
        else:
            self.model = self.get_model(**self.kwargs)
        if Xt is None or yt is None:
            self.model.fit(Xs, ys, **fit_params)
        else:
            self._fit(Xs, ys, Xt, yt, **fit_params)
        return self
    
    
    def predict(self, X):
        return self.model.predict(X)

    
class SourceOnly(TrainingModel):
            
    def _fit(self, Xs, ys, Xt, yt, **fit_params):
        self.model.fit(Xs, ys, **fit_params)
        return self
    
    
class TargetOnly(TrainingModel):
            
    def _fit(self, Xs, ys, Xt, yt, **fit_params):
        self.model.fit(Xt, yt, **fit_params)
        return self
    
    
    
class UniformWeighting(TrainingModel):
            
    def _fit(self, Xs, ys, Xt, yt, **fit_params):
        X = np.concatenate((Xs, Xt))
        y = np.concatenate((ys, yt))
        self.model.fit(X, y, **fit_params)
        return self
    

class BalanceWeighting(TrainingModel):
        
    def _fit(self, Xs, ys, Xt, yt, **fit_params):
        
        n_s = len(Xs)
        n_t = len(Xt)
        sample_weight = np.concatenate((np.ones(n_s), np.sqrt(n_s / n_t) * np.ones(n_t)))
        sample_weight /= sample_weight.sum()

        X = np.concatenate((Xs, Xt))
        y = np.concatenate((ys, yt))

        if "sample_weight" in inspect.signature(self.model.fit).parameters:
            self.model.fit(X, y,
                           sample_weight=sample_weight,
                           **fit_params)
        else:
            train_index = np.random.choice(len(X),
                                           size=len(X),
                                           replace=True,
                                           p=sample_weight)
            self.model.fit(X[train_index], y[train_index],
                           **fit_params)
        return self