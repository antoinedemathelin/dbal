"""
Training models
"""

import inspect

import numpy as np


class TrainingModel:
    
    def __init__(self, get_model, **kwargs):
        self.get_model = get_model
        self.kwargs = kwargs
        
    def fit(self, X, y, src_index=None, tgt_index=None, **fit_params):
        
        if "input_shape" in inspect.signature(self.get_model).parameters:
            self.model = self.get_model(input_shape=X.shape[1:],
                                        output_shape=y.shape[1:],
                                        **self.kwargs)
        else:
            self.model = self.get_model(**self.kwargs)
        if src_index is None or tgt_index is None:
            self.model.fit(X, y, **fit_params)
        else:
            self._fit(X, y, src_index, tgt_index, **fit_params)
        return self
    
    
    def predict(self, X):
        return self.model.predict(X)

    
class SourceOnly(TrainingModel):
            
    def _fit(self, X, y, src_index, tgt_index, **fit_params):
        self.model.fit(X[src_index], y[src_index], **fit_params)
        return self
    
    
class TargetOnly(TrainingModel):
            
    def _fit(self, X, y, src_index, tgt_index, **fit_params):
        self.model.fit(X[tgt_index], y[tgt_index], **fit_params)
        return self
    
    
    
class UniformWeighting(TrainingModel):
            
    def _fit(self, X, y, src_index, tgt_index, **fit_params):
        train_index = np.concatenate((src_index, tgt_index))
        self.model.fit(X[train_index], y[train_index], **fit_params)
        return self
    

class BalanceWeighting(TrainingModel):
        
    def _fit(self, X, y, src_index, tgt_index, **fit_params):
        
        n_s = len(src_index)
        n_t = len(tgt_index)
        sample_weight = np.concatenate((np.ones(n_s), np.sqrt(n_s / n_t) * np.ones(n_t)))
        sample_weight /= sample_weight.sum()

        train_index = np.concatenate((src_index, tgt_index))

        if "sample_weight" in inspect.signature(self.model.fit).parameters:
            self.model.fit(X[train_index], y[train_index],
                           sample_weight=sample_weight,
                           **fit_params)
        else:
            train_index = np.random.choice(train_index,
                                           size=len(train_index),
                                           replace=True,
                                           p=sample_weight)
            self.model.fit(X[train_index], y[train_index],
                           **fit_params)
        return self