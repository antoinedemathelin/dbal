"""
Kmedoids Algorithm
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




class MiniBatchKMedoids:
    
    def __init__(self, distance=manhattan_distances,
                 nn_algorithm="kdt-forest", n_trees=50, leaf_size="log",
                 n_samples=10, batch_size_init=3000,
                 max_iter=10, tol=1e-4, batch_size=100, verbose=0):
        self.distance = distance
        self.nn_algorithm = nn_algorithm
        self.n_trees = n_trees
        self.leaf_size = leaf_size
        self.n_samples = n_samples
        self.batch_size_init = batch_size_init
        self.max_iter = max_iter
        self.tol = tol
        self.batch_size = batch_size
        self.verbose = verbose
        self.distance_to_nearest_src = None
        
    def fit_nearest_neighbours(self, Xs, Xt):       
        if self.nn_algorithm == "brute":
            self.distance_to_nearest_src = self.distance(Xt, Xs).min(1)
            
        elif self.nn_algorithm == "kdt-forest":
            kdtf = KDTreeForest(n_trees=self.n_trees,
                          leaf_size=self.leaf_size,
                          distance=self.distance)
            if self.verbose:
                print("Fitting KDT-Forest...")
                validation_set = Xt[np.random.choice(len(Xt), 100, replace=False)]
                kdtf.fit(Xs, validation_set=validation_set)
                print("KDT-Forest predict of Nearest Neighour distances...")
            else:
                kdtf.fit(Xs)
            self.distance_to_nearest_src = kdtf.predict(Xt)

        else:
            raise ValueError("nn_algorithm should be kdt-forest or brute")
    
    
    def fit(self, Xt, distances_to_src=None):
        if distances_to_src is not None:
            self.distance_to_nearest_src = distances_to_src
        else:
            if self.distance_to_nearest_src is None:
                self.distance_to_nearest_src = np.full(len(Xt), np.inf)
        self.select_indexes = np.random.choice(len(Xt), self.batch_size_init, replace=False)
        queries = self._initialization(Xt[self.select_indexes])
        self._kmedoids(Xt, queries)
    
    
    def _initialization(self, Xt):
        if self.verbose:
            print("Computation distance initial batch...")
        distance_matrix_initial_batch = self.distance(Xt)

        self.queries = []
        for i in range(self.n_samples):
            deltas = np.min(np.stack(
                [self.distance_to_nearest_src[self.select_indexes]] +
                [distance_matrix_initial_batch[i] for i in self.queries],
                axis=1), axis=1)
            deltas_matrix = np.resize(deltas, distance_matrix_initial_batch.shape)
            diff_matrix = np.clip(deltas_matrix - distance_matrix_initial_batch, a_min=0., a_max=None)
            self.queries.append(diff_matrix.dot(np.ones((len(Xt), 1))).ravel().argmax())

            if self.verbose:
                print("Queries number: %i --- Objective: %.3f"%(len(self.queries), deltas.mean()))
                
        return self.queries
            
        
    def _kmedoids(self, Xt, queries):
        self.centers_index = self.select_indexes[queries]
        self.distance_to_centers = self.distance(Xt, Xt[self.centers_index])

        self.cluster_indexes = np.argmin(np.concatenate((
                self.distance_to_nearest_src.reshape(-1, 1),
                self.distance_to_centers),
                axis=1), axis=1) - 1

        self.distance_to_nearest_center = np.min(np.concatenate((
                self.distance_to_nearest_src.reshape(-1, 1),
                self.distance_to_centers),
                axis=1), axis=1)

        previous_objective = np.inf
        self.objective = self.distance_to_nearest_center.mean()
        epoch = 0

        while previous_objective - self.objective > self.tol and epoch < self.max_iter:

            if self.verbose:
                print("Epoch %i -- Objective: %.3f"%(epoch, self.objective))
            
            change_mask = np.full(len(self.centers_index), False)
            for k in range(self.n_samples):
                best_mean = self.distance_to_nearest_center[self.cluster_indexes == k].mean()
                argmin = self._compute_cluster_sums(Xt[self.cluster_indexes == k], k, best_mean)
                if argmin is not None:
                    self.centers_index[k] = np.argwhere(self.cluster_indexes == k).ravel()[argmin]
                    change_mask[k] = True
            
            if change_mask.sum() >= 1:
                self.distance_to_centers[:, change_mask] = self.distance(Xt, Xt[self.centers_index[change_mask]])

                self.cluster_indexes = np.argmin(np.concatenate((
                        self.distance_to_nearest_src.reshape(-1, 1),
                        self.distance_to_centers),
                        axis=1), axis=1) - 1

                self.distance_to_nearest_center = np.min(np.concatenate((
                        self.distance_to_nearest_src.reshape(-1, 1),
                        self.distance_to_centers),
                        axis=1), axis=1)

            previous_objective = copy.deepcopy(self.objective)
            self.objective = self.distance_to_nearest_center.mean()
            epoch += 1
            
    
    def _compute_cluster_sums(self, Xt_k, k, best_mean):
        batch_size = self.batch_size
        batch_indexes = np.random.choice(len(Xt_k),
                                         len(Xt_k),
                                         replace=False)
        keep_indexes = np.full(len(Xt_k), True)

        i = 0
        cluster_distances = None
        thresholds = [best_mean, np.inf, np.inf]

        while batch_size * i < len(Xt_k) and keep_indexes.sum() > batch_size:

            if i == 0:
                cluster_distances = self.distance(Xt_k[keep_indexes],
                                                       Xt_k[batch_indexes[batch_size * i : batch_size * (i+1)]])
                thresholds[1] = np.mean(cluster_distances, 0).min()
                argmin_first_batch = batch_indexes[np.argmin(np.mean(cluster_distances, 0))]
            else:
                cluster_distances = np.concatenate((
                    cluster_distances,
                    self.distance(Xt_k[keep_indexes],
                                Xt_k[batch_indexes[batch_size * i : batch_size * (i+1)]])
                ), axis=1)

            cluster_mean = np.mean(cluster_distances, 1)
            cluster_std = np.std(cluster_distances, 1)
            argmin = np.argmin(cluster_mean)
            thresholds[2] = (cluster_mean + (2 * cluster_std) / np.sqrt(batch_size * (i+1)))[argmin]

            keep_mask = cluster_mean - (2 * cluster_std) / np.sqrt(batch_size * (i+1)) < np.min(thresholds)

            cluster_distances = cluster_distances[keep_mask]
            keep_indexes[np.argwhere(keep_indexes).ravel()] = keep_mask

            i += 1

        if batch_size * i < len(Xt_k):
            if keep_indexes.sum() >= 1:
                if i == 0:
                    cluster_distances = self.distance(Xt_k[keep_indexes],
                                               Xt_k[batch_indexes[batch_size * i : ]])
                else:
                    cluster_distances = np.concatenate((
                            cluster_distances,
                            self.distance(Xt_k[keep_indexes],
                                               Xt_k[batch_indexes[batch_size * i : ]])
                        ), axis=1)
                cluster_mean = np.mean(cluster_distances, 1)
                thresholds[2] = cluster_mean.min()
                argmin_cluster_mean = np.argwhere(keep_indexes).ravel()[np.argmin(cluster_mean)]
            else:
                thresholds[2] = np.inf

        else:
            thresholds[2] = cluster_mean.min()
            argmin_cluster_mean = np.argwhere(keep_indexes).ravel()[np.argmin(cluster_mean)]

        thresholds_argmin = np.argmin(thresholds)
        
        if thresholds_argmin == 1:
            return argmin_first_batch
        elif thresholds_argmin == 2:
            return argmin_cluster_mean
        else:
            return None