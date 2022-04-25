"""
Active Learning Methods
"""

import copy

import numpy as np
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, MiniBatchKMeans
from kdtrees import KDTreeForest
import tensorflow.keras.backend as K


def qbc_uncertainties(y_preds):
    predictions = np.stack(y_preds, axis=-1)
    uncertainties = np.std(predictions, axis=-1).ravel()
    return uncertainties


def aada_uncertainties(y_pred, y_disc):
    if len(y_pred.shape) == 1 or y_pred.shape[1] == 1:
        y_pred = y_pred.ravel()
        uncertainties = -((1-y_pred) * np.log(1-y_pred + 1e-6) +
                       y_pred * np.log(y_pred + 1e-6))
    else:
        uncertainties = y_pred * np.log(y_pred + 1e-6)
        uncertainties = -np.sum(uncertainties, axis=1).ravel()

    importance = y_disc/(1-y_disc + 1e-6)
    return importance * uncertainties


def bvsb_uncertainties(y_pred):
    if len(y_pred.shape) == 1 or y_pred.shape[1] == 1:
        y_pred = y_pred.ravel()
        uncertainties = 0.5 - np.abs(0.5 - y_pred)
    else:
        sorted_pred = np.sort(y_pred, axis=1)
        uncertainties = 1 - (sorted_pred[:, -1] - sorted_pred[:, -2])
        uncertainties = uncertainties.ravel()
    return uncertainties


class BaseQuery:
    
    def __init__(self, model=None):
        self.model = model
        self.queries = []


    def fit(self, Xt, Xs=None, ys=None, n_queries=None,
            sample_weight=None):
        
        self.Xt_ = Xt
        self.Xs_ = Xs
        self.ys_ = ys
        self.sample_weight_ = sample_weight
        
        if n_queries is None:
            n_queries = 1
        if n_queries > len(Xt):
            n_queries = len(Xt)
        if sample_weight is None:
            sample_weight = np.ones(len(Xt))
        
        self._fit(Xt=Xt, Xs=Xs, ys=ys,
                  n_queries=n_queries,
                  sample_weight=sample_weight)
        return self
    
    
    def predict(self, n_queries=None):
        if n_queries is None:
            n_queries = len(self.queries)
        return self._predict(Xt=self.Xt_, Xs=self.Xs_, ys=self.ys_,
                      n_queries=n_queries,
                      sample_weight=self.sample_weight_)
        
    
    
    def fit_predict(self, Xt, Xs=None, ys=None, n_queries=None,
            sample_weight=None):
        self.fit(Xt=Xt, Xs=Xs, ys=ys,
                  n_queries=n_queries,
                  sample_weight=sample_weight)
        return self.predict(n_queries)
        
    
    def _fit(self, Xt, Xs, ys, n_queries,
            sample_weight):
        pass
    
    def _predict(self, Xt, Xs, ys, n_queries,
                sample_weight):
        return self.queries[:n_queries]
    
    
class OrderedQuery(BaseQuery):
    
    def _fit(self, Xt, Xs, ys, n_queries,
                sample_weight):
        if sample_weight is None:
            sample_weight = np.ones(len(Xt))
        self.queries = np.argsort(sample_weight)[::-1]
        return self
    
    
class KMedoidsAccelerated(BaseQuery):
    
    def __init__(self, verbose=0,
                 distance=manhattan_distances,
                 nn_algorithm="kdt-forest",
                 batch_size_init=3000,
                 max_iter=10, tol=1e-4,
                 batch_size=100, **kwargs):
        super().__init__()
        self.distance = distance
        self.nn_algorithm = nn_algorithm
        self.batch_size_init = batch_size_init
        self.max_iter = max_iter
        self.tol = tol
        self.batch_size = batch_size
        self.verbose = verbose
        self.kwargs = kwargs
        self.distance_to_nearest_src = None
        
        
    def _predict(self, Xt, Xs, ys, n_queries,
                sample_weight):
        return self._kmedoids(Xt, self.queries[:n_queries])
        
    
    def _fit(self, Xt, Xs, ys, n_queries, sample_weight, **fit_params):
        self._fit_nearest_neighbours(Xs, Xt)
        batch_size_init = min(len(Xt), self.batch_size_init)
        self.select_indexes = np.random.choice(len(Xt),
                                               batch_size_init,
                                               replace=False)
        self.weights = sample_weight
        if sample_weight is None:
            sample_weight = np.ones(len(Xt))
        queries = self._initialization(Xt[self.select_indexes],
                                       n_queries,
                                       self.distance_to_nearest_src[self.select_indexes],
                                       sample_weight[self.select_indexes])
        self.queries = self.select_indexes[queries]
        return self


    def _fit_nearest_neighbours(self, Xs, Xt):
        if Xs is None:
            self.distance_to_nearest_src = np.full((len(Xs),), np.inf)
        else:
            if self.nn_algorithm == "brute":
                self.distance_to_nearest_src = self.distance(Xt, Xs).min(1)
            elif self.nn_algorithm == "kdt-forest":
                kdtf = KDTreeForest(distance=self.distance, **self.kwargs)
                if self.verbose > 1:
                    print("Fitting KDT-Forest...")
                    validation_set = Xt[np.random.choice(len(Xt), 100, replace=False)]
                    kdtf.fit(Xs, validation_set=validation_set)
                    print("KDT-Forest predict of Nearest Neighour distances...")
                else:
                    kdtf.fit(Xs)
                self.distance_to_nearest_src = kdtf.predict(Xt)
            else:
                raise ValueError("nn_algorithm should be 'kdt-forest' or 'brute'")


    def _initialization(self, Xt, n_queries, distance_to_nearest_src, weights):
        if self.verbose:
            print("Computation distance initial batch...")
        distance_matrix_initial_batch = self.distance(Xt)
        queries = []
        for i in range(n_queries):
            deltas = np.min(np.stack(
                [distance_to_nearest_src] +
                [distance_matrix_initial_batch[i] for i in queries],
                axis=1), axis=1)
            deltas_matrix = np.resize(deltas, distance_matrix_initial_batch.shape)
            diff_matrix = np.clip(deltas_matrix - distance_matrix_initial_batch, a_min=0., a_max=None)
            queries.append(diff_matrix.dot(weights.reshape(-1, 1)).ravel().argmax())

            if self.verbose == 1:
                if i % int(n_queries/10) == 0:
                    print("Queries number: %i --- Objective: %.3f"%(len(queries), deltas.mean()))
            if self.verbose > 1:
                print("Queries number: %i --- Objective: %.3f"%(len(queries), deltas.mean()))
        return queries
            
        
    def _kmedoids(self, Xt, queries):
        self.centers_index = queries
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
        if self.weights is None:
            self.objective = self.distance_to_nearest_center.mean()
        else:
            self.objective = (self.distance_to_nearest_center.ravel() * self.weights.ravel()).mean()
        epoch = 0

        while previous_objective - self.objective > self.tol and epoch < self.max_iter:

            print("Epoch %i -- Objective: %.3f"%(epoch, self.objective))
            
            change_mask = np.full(len(self.centers_index), False)
            for k in range(len(self.centers_index)):
                if self.weights is None:
                    best_mean = self.distance_to_nearest_center[self.cluster_indexes == k].mean()
                else:
                    best_mean = (self.distance_to_nearest_center[self.cluster_indexes == k].ravel() *
                                 self.weights[self.cluster_indexes == k].ravel()).mean()
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
            if self.weights is None:
                self.objective = self.distance_to_nearest_center.mean()
            else:
                self.objective = (self.distance_to_nearest_center.ravel() * self.weights.ravel()).mean()
            epoch += 1
            
        return self.centers_index
            
    
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
                if self.weights is None:
                    thresholds[1] = np.mean(cluster_distances, 0).min()
                    argmin_first_batch = batch_indexes[np.argmin(np.mean(cluster_distances, 0))]
                else:
                    weights_vector = self.weights[self.cluster_indexes == k].reshape(-1, 1)
                    weights_matrix = np.repeat(weights_vector, cluster_distances.shape[1], 1)
                    thresholds[1] = np.mean(cluster_distances * weights_matrix, 0).min()
                    argmin_first_batch = batch_indexes[np.mean(cluster_distances * weights_matrix, 0).argmin()]
            else:
                cluster_distances = np.concatenate((
                    cluster_distances,
                    self.distance(Xt_k[keep_indexes],
                                Xt_k[batch_indexes[batch_size * i : batch_size * (i+1)]])
                ), axis=1)
            
            if self.weights is None:
                cluster_mean = np.mean(cluster_distances, 1)
                cluster_std = np.std(cluster_distances, 1)
            else:
                weights_vector = self.weights[self.cluster_indexes == k][batch_indexes[: batch_size * (i+1)]].reshape(1, -1)
                weights_matrix = np.repeat(weights_vector, len(cluster_distances), 0)
                cluster_mean = np.mean(cluster_distances * weights_matrix, 1)
                cluster_std = np.std(cluster_distances * weights_matrix, 1)
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
                if self.weights is None:
                    cluster_mean = np.mean(cluster_distances, 1)
                else:
                    weights_vector = self.weights[self.cluster_indexes == k][batch_indexes].reshape(1, -1)
                    weights_matrix = np.repeat(weights_vector, len(cluster_distances), 0)
                    cluster_mean = np.mean(cluster_distances * weights_matrix, 1)
                thresholds[2] = cluster_mean.min()
                argmin_cluster_mean = np.argwhere(keep_indexes).ravel()[np.argmin(cluster_mean)]
            else:
                thresholds[2] = np.inf

        else:
            if keep_indexes.sum() >= 1:
                if self.weights is None:
                    cluster_mean = np.mean(cluster_distances, 1)
                else:
                    weights_vector = self.weights[self.cluster_indexes == k][batch_indexes].reshape(1, -1)
                    weights_matrix = np.repeat(weights_vector, len(cluster_distances), 0)
                    cluster_mean = np.mean(cluster_distances * weights_matrix, 1)
                thresholds[2] = cluster_mean.min()
                argmin_cluster_mean = np.argwhere(keep_indexes).ravel()[np.argmin(cluster_mean)]
            else:
                thresholds[2] = np.inf

        if self.verbose > 1:
            print(k, batch_size * i, keep_indexes.sum(), thresholds)
        thresholds_argmin = np.argmin(thresholds)
        
        if thresholds_argmin == 1:
            return argmin_first_batch
        elif thresholds_argmin == 2:
            return argmin_cluster_mean
        else:
            return None
    

    
class KMedoidsQuery(BaseQuery):
    
    def __init__(self, distance=manhattan_distances,
                 nn_algorithm="brute", **kwargs):
        self.distance = distance
        self.nn_algorithm = nn_algorithm
        self.kwargs = kwargs
        super().__init__()


    def _fit(self, Xt, Xs, ys, n_queries, sample_weight):
        if Xs is None:
            deltas = np.full((len(Xt),), np.inf)
        else:
            if self.nn_algorithm == "brute":
                distance_matrix_src = self.distance(Xt, Xs)
                deltas_src = np.min(distance_matrix_src, axis=1).ravel()
            elif self.nn_algorithm == "kd-trees":
                self.kdtf = KDTreeForest(distance=self.distance, **self.kwargs)
                self.kdtf.fit(Xs)
                deltas_src = self.kdtf.predict(Xt)
            else:
                raise ValueError("nn_algorithm should be None, 'kdt-forest' or 'brute'")

        distance_matrix_tgt = self.distance(Xt)
        
        self.queries = []
        
        for i in range(n_queries):
            self.deltas = np.min(np.stack(
                [deltas_src] + [distance_matrix_tgt[i] for i in self.queries],
                axis=1), axis=1)
            deltas_matrix = np.resize(self.deltas, distance_matrix_tgt.shape)
            diff_matrix = np.clip(deltas_matrix - distance_matrix_tgt, a_min=0., a_max=None)
            self.queries.append(diff_matrix.dot(sample_weight.reshape(-1, 1)).ravel().argmax())
            
        self.deltas = np.min(np.stack(
                [deltas_src] + [distance_matrix_tgt[i] for i in self.queries],
                axis=1), axis=1)
        return self


class KMeansQuery(BaseQuery):
    
    def __init__(self, minibatch=False):
        self.minibatch = minibatch
        super().__init__()
    
    def _predict(self, Xt, Xs, ys, n_queries, sample_weight):
        if sample_weight is not None:
            sample_weight /= np.sum(sample_weight)
        if self.minibatch:
            kmeans = MiniBatchKMeans(n_clusters=n_queries)
        else:
            kmeans = KMeans(n_clusters=n_queries)
        kmeans.fit(Xt, sample_weight=sample_weight)
        centers = kmeans.cluster_centers_
        nearest = NearestNeighbors(n_neighbors=1)
        nearest.fit(Xt)
        dist, index = nearest.kneighbors(centers, 1)
        return list(index.ravel())


class RandomQuery(BaseQuery):

    def _fit(self, Xt, Xs, ys, n_queries, sample_weight):
        assert np.all(sample_weight >= 0)
        assert np.sum(sample_weight) != 0
        sample_weight /= np.sum(sample_weight)
        self.queries = np.random.choice(len(Xt),
                                        n_queries,
                                        replace=False,
                                        p=sample_weight)
        return self


class KCentersQuery(BaseQuery):
    
    def __init__(self, distance=manhattan_distances,
                 nn_algorithm="brute", **kwargs):
        self.distance = distance
        self.nn_algorithm = nn_algorithm
        self.kwargs = kwargs
        super().__init__()
    
    
    def _fit(self, Xt, Xs, ys, n_queries, sample_weight):
        if sample_weight is not None:
            sample_weight /= sample_weight.sum()
        if Xs is None:
            deltas = np.full((len(Xt),), np.inf)
        else:
            if self.nn_algorithm == "brute":
                distance_matrix_src = self.distance(Xt, Xs)
                deltas = np.min(distance_matrix_src, axis=1).ravel()
            elif self.nn_algorithm == "kdt-forest":
                self.kdtf = KDTreeForest(distance=self.distance, **self.kwargs)
                self.kdtf.fit(Xs)
                deltas = self.kdtf.predict(Xt)
            else:
                raise ValueError("nn_algorithm should be None, 'kdt-forest' or 'brute'")
        
        self.queries = []
        for i in range(n_queries):
            if sample_weight is None:
                argmax = deltas.argmax()
            else:
                argmax = (deltas * sample_weight).argmax()
            self.queries.append(argmax)
            deltas_tgt = self.distance(Xt,
                                       Xt[[argmax]])
            deltas = np.min(np.concatenate((deltas.reshape(-1, 1),
                                            deltas_tgt.reshape(-1, 1)),
                                           axis=1), axis=1).ravel()
        return self  
    
    
class DiversityQuery(BaseQuery):
    
    def __init__(self, distance=manhattan_distances,
                 nn_algorithm="brute", **kwargs):
        self.distance = distance
        self.nn_algorithm = nn_algorithm
        self.kwargs = kwargs
        super().__init__()

    
    def _fit(self, Xt, Xs, ys, n_queries, sample_weight):
        if Xs is None:
            deltas = np.full((len(Xt),), np.inf)
        else:
            if self.nn_algorithm == "brute":
                distance_matrix_src = self.distance(Xt, Xs)
                deltas = np.mean(distance_matrix_src, axis=1).ravel()
            elif self.nn_algorithm == "kdt-forest":
                self.kdtf = KDTreeForest(distance=self.distance, **self.kwargs)
                self.kdtf.fit(Xs)
                deltas = self.kdtf.predict(Xt)
            else:
                raise ValueError("nn_algorithm should be None, 'kdt-forest' or 'brute'")
        
        self.queries = np.argsort(deltas).ravel()[::-1]
        return self

        
class AADA:
    
    def __init__(self, y_pred, y_disc):
        self.y_pred = y_pred
        self.y_disc = y_disc
    
    def uncertainties(self, X):
        if len(self.y_pred.shape) == 1 or self.y_pred.shape[1] == 1:
            y_pred = self.y_pred.ravel()
            uncertainties = -((1-y_pred) * np.log(1-y_pred + 1e-6) +
                           y_pred * np.log(y_pred + 1e-6))
        else:
            y_pred = self.y_pred
            uncertainties = y_pred * np.log(y_pred + 1e-6)
            uncertainties = -np.sum(uncertainties, axis=1).ravel()
        
        importance = self.y_disc/(1-self.y_disc + 1e-6)
        return importance * uncertainties
    

class BVSB:
    
    def __init__(self, get_base_model, **kwargs):
        self.model = get_base_model(**kwargs)

    def fit(self, X, y, **fit_params):
        self.model.fit(X, y, **fit_params)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def uncertainties(self, X):
        y_pred = self.model.predict(X)
        if len(y_pred.shape) == 1 or y_pred.shape[1] == 1:
            y_pred = y_pred.ravel()
            uncertainties = 0.5 - np.abs(0.5 - y_pred)
        else:
            sorted_pred = np.sort(y_pred, axis=1)
            uncertainties = 1 - (sorted_pred[:, -1] - sorted_pred[:, -2])
            uncertainties = uncertainties.ravel()
        return uncertainties
        
    def embeddings(self, X, layer=-2):
        if isinstance(layer, str):
            embeddings = K.function([self.model.layers[0].input],
                [self.model.get_layer(layer).output])(X)[0]
        else:
            embeddings = K.function([self.model.layers[0].input],
                [self.model.get_layer(index=layer).output])(X)[0]
        return embeddings
    
    
class QBC:
    
    def __init__(self, get_base_model, n_models=10, **kwargs):
        self.n_models = n_models
        self.models = [get_base_model(**kwargs) for _ in range(self.n_models)]
        
    
    def fit(self, X, y, **fit_params):
        predictions = []
        for model in self.models:
            index_boot = np.random.choice(len(X),
                                          size=len(X),
                                          replace=True)
            model.fit(X[index_boot], y[index_boot], **fit_params)
        return self
    
    def predict(self, X):
        predictions = np.stack([model.predict(X) for model in self.models], axis=-1)
        return np.mean(predictions, axis=-1)
    
    def uncertainties(self, X):
        predictions = np.stack([model.predict(X) for model in self.models], axis=-1)
        uncertainties = np.std(predictions, axis=-1).ravel()
        return uncertainties
        
    def embeddings(self, X, layer=-2):
        if isinstance(layer, str):
            embeddings = K.function([self.models[0].layers[0].input],
                [self.models[0].get_layer(layer).output])(X)[0]
        else:
            embeddings = K.function([self.models[0].layers[0].input],
                [self.models[0].get_layer(index=layer).output])(X)[0]
        return embeddings
