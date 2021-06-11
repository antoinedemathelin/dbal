"""
KD-Trees Algorithm
"""

import copy

import numpy as np
from sklearn.metrics.pairwise import manhattan_distances


class KDTree:
    
    def __init__(self, features_order="var",
                 leaf_size="sqrt",
                 distance=manhattan_distances):
        self.features_order = features_order
        self.leaf_size = leaf_size
        self.distance = distance
        
    def fit(self, X):
        self.X = X
        
        if self.features_order == "var":
            variance_of_features = self.X.var(axis=0)
            sorted_features = np.argsort(variance_of_features).ravel()[::-1]
            
        elif self.features_order == "random":
            sorted_features = np.random.choice(self.X.shape[1],
                                               self.X.shape[1],
                                               replace=False)
        else:
            raise ValueError("feature_order sould be var or random")
            
        if self.leaf_size == "sqrt":
            self.max_branch = int(np.log2(len(self.X)) / (2 * np.log2(2)))
        elif self.leaf_size == "log":
            self.max_branch = int(np.log2(len(self.X) / np.log2(len(self.X))) / np.log2(2))
        else:
            self.max_branch = int(np.log2(len(self.X) / self.leaf_size) / np.log2(2))
        
        self.features = sorted_features[:self.max_branch]

        self.clusters = []

        tree = {"args": np.array(range(len(self.X)), dtype=np.int32)}
        self.tree = self._recursion(tree, 0)
        
    
    def _split(self, args, i):
        x = self.X[args, self.features[i]]
        argsort_x = np.argsort(x).ravel()
        argsort = args[argsort_x]
        split = int(len(args)/2)
        return (argsort[:split], argsort[split:], x[split])
        
    def _add_leaf(self, tree, i):
        splited_args = self._split(tree["args"], i)
        tree[str(i) + "_left"] = {"args": splited_args[0]}
        tree[str(i) + "_right"] = {"args": splited_args[1]}
        tree["feature"] = self.features[i]
        tree["threshold"] = splited_args[2]
        del tree["args"]
        return tree
    
    def _recursion(self, tree, i):       
        if i == self.max_branch:
            self.clusters.append(tree["args"])
            return tree
        elif "args" in tree.keys():
            if len(tree["args"]) == 0:
                return tree
            elif len(tree["args"]) == 1:
                self.clusters.append(tree["args"])
                return tree
            else:
                return self._recursion(self._add_leaf(tree, i), i)
        else:
            tree[str(i) + "_left"] = self._recursion(tree[str(i) + "_left"], i+1)
            tree[str(i) + "_right"] = self._recursion(tree[str(i) + "_right"], i+1)
        return tree


    def _split_predict(self, args, i, threshold):
        x = self.X_predict[args, self.features[i]]
        args_left = args[np.argwhere(x < threshold).ravel()]
        args_right = args[np.argwhere(x >= threshold).ravel()]
        return (args_left, args_right)
    
    
    def _neighbour(self, args_predict, args_fit):
        distances = self.distance(self.X_predict[args_predict],
                                  self.X[args_fit])
        min_distances = np.min(distances, axis=1)
        new_min_args = np.argwhere(self.distance_predict[args_predict] > min_distances)
        self.distance_predict[args_predict[new_min_args]] = min_distances[new_min_args]
    
    
    def _recursion_predict(self, tree, args, i):
        if "args" in tree.keys():
            self._neighbour(args, tree["args"])
        else:
            threshold = tree["threshold"]
            splited_args = self._split_predict(args, i, threshold)
            if len(splited_args[0]) > 0:
                self._recursion_predict(tree[str(i) + "_left"], splited_args[0], i+1)
            if len(splited_args[1]) > 0:
                self._recursion_predict(tree[str(i) + "_right"], splited_args[1], i+1)
            
    
    def predict(self, X):
        self.X_predict = X
        self.distance_predict = np.full(len(self.X_predict), np.inf)
        args = np.array(range(len(self.X_predict)), dtype=np.int32)
        self._recursion_predict(self.tree, args, 0)
        return self.distance_predict



class KDTreeForest:

    def __init__(self, n_trees=20, leaf_size="log",
                 distance=manhattan_distances):
        self.n_trees = n_trees
        self.leaf_size = leaf_size
        self.distance = distance
        
    
    def fit(self, X, validation_set=None):
        
        if validation_set is not None:
            true_min_distance = self.distance(validation_set, X).min(1)
            history = []
        
        self.trees = []
        
        for k in range(self.n_trees):
            self.trees.append(KDTree(features_order="random", leaf_size=self.leaf_size, distance=self.distance))
            self.trees[-1].fit(X)
            
            if validation_set is not None:
                history.append(self.trees[-1].predict(validation_set).ravel())
                predict_min_distance = np.stack(history).min(0)
                errors = np.abs(true_min_distance - predict_min_distance)
                print("Mean Absolute Error on Distance: %.3f (%.3f)"%(np.mean(errors), np.std(errors)))
        
        return self
    
    
    def predict(self, X):
        predict_min_distances = np.stack([self.trees[i].predict(X).ravel() 
                                          for i in range(len(self.trees))])
        return predict_min_distances.min(0)