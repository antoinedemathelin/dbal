"""
Active Learning Methods
"""

import copy

import numpy as np
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, MiniBatchKMeans
from kmedoids import MiniBatchKMedoids, KDTreeForest, KDTree


def kmeans(X, n_samples, weights=None, minibatch=False):
    print("minibatch %s"%str(minibatch))
    if minibatch:
        kmeans = MiniBatchKMeans(n_clusters=n_samples)
    else:
        kmeans = KMeans(n_clusters=n_samples)
    kmeans.fit(X, sample_weight=weights)
    centers = kmeans.cluster_centers_
    nearest = NearestNeighbors(n_neighbors=1)
    nearest.fit(X)
    dist, index = nearest.kneighbors(centers, 1)
    return index.ravel()


class BaseQueryInformative:
    def __init__(self, minibatch=False):
        self.X_ = None
        self.information_ = None
        self.n_samples_ = None
        self.minibatch = minibatch


    def fit(self, X, y, src_index, tgt_index, n_samples, **fit_params):
        self.X = X[tgt_index]
        return self
    
    
    def predict(self, n_samples=None):
        assert self.X_ is not None
        assert self.information_ is not None
        assert len(self.X_) == len(self.information_)
        
        if n_samples is None:
            n_samples = self.n_samples_
            
        assert n_samples <= len(self.X_)
            
        weights = self.information_/self.information_.sum()
        
        return kmeans(self.X_, n_samples, weights, self.minibatch)


class BaseQueryRepresentative:
    def __init__(self):
        self.indexes_ = None
        self.n_samples_ = None


    def fit(self, X, y, src_index, tgt_index, n_samples, **fit_params):
        pass
    
    
    def predict(self, n_samples=None):
        assert self.indexes_ is not None
        
        if n_samples is None:
            n_samples = self.n_samples_
            
        assert n_samples <= len(self.indexes_)
        
        return self.indexes_[:n_samples]

    
class MinDiscrepancyGreedy(BaseQueryRepresentative):
    
    def __init__(self, get_model, loss_fct, cut_fct=None,
                 distance=manhattan_distances, verbose=0, **kwargs):
        self.get_model = get_model
        self.loss_fct = loss_fct
        self.cut_fct = cut_fct
        self.distance = distance
        self.verbose = verbose
        self.kwargs = kwargs
        super().__init__()
    
    
    def _get_lipschitz_constant(self, X, y, Xt=None, yt=None):
        if Xt is None or yt is None:
            dist_X = self.distance(X)
            dist_y = self.distance(y)
        else:
            dist_X = self.distance(X, Xt)
            dist_y = self.distance(y, yt)
        return np.max(dist_y / (dist_X + 1e-5))
    
    
    def fit(self, X, y, src_index, tgt_index, n_samples, **fit_params):
        """
        """
        self.model = self.get_model(**self.kwargs, input_shape=X.shape[1:], output_shape=y.shape[1:])
        self.model.fit(X[src_index], y[src_index], **fit_params)
        
        if self.cut_fct is not None:
            y_pred = self.cut_fct(self.model)(X)
        else:
            y_pred = self.model.predict(X)
        
        queries = []
                
        if len(y_pred.shape) <= 1:
            y_pred = y_pred.reshape(-1, 1)

        lipschitz_constant_src = self._get_lipschitz_constant(X[src_index], y_pred[src_index])
        lipschitz_constant_tgt = self._get_lipschitz_constant(X[tgt_index], y_pred[tgt_index])
        lipschitz_constant_src_tgt = self._get_lipschitz_constant(X[src_index], y_pred[src_index], X[tgt_index], y_pred[tgt_index])
        lipschitz_constant = np.dstack((lipschitz_constant_src, lipschitz_constant_tgt, lipschitz_constant_src_tgt)).max(-1)
        
        if self.verbose:
            print("Lipschitz Constant: %.3f"%lipschitz_constant)

        distance_matrix_src = self.distance(X[tgt_index], X[src_index])
        f_matrix_src = np.resize(y_pred[src_index], (len(tgt_index),) + y_pred[src_index].shape)

        distance_matrix_tgt = self.distance(X[tgt_index])
        f_matrix_tgt = np.resize(y_pred[tgt_index], (len(tgt_index),) + y_pred[tgt_index].shape)

        resize_distance_matrix_src = np.repeat(distance_matrix_src[:, :, np.newaxis], f_matrix_src.shape[2], axis=2)
        resize_distance_matrix_tgt = np.repeat(distance_matrix_tgt[:, :, np.newaxis], f_matrix_tgt.shape[2], axis=2)

        d_matrix_plus_src = f_matrix_src + lipschitz_constant * resize_distance_matrix_src
        d_matrix_minus_src = f_matrix_src - lipschitz_constant * resize_distance_matrix_src

        d_matrix_plus_tgt = f_matrix_tgt + lipschitz_constant * resize_distance_matrix_tgt
        d_matrix_minus_tgt = f_matrix_tgt - lipschitz_constant * resize_distance_matrix_tgt

        d_plus_src = d_matrix_plus_src.min(1)
        d_minus_src = d_matrix_minus_src.max(1)

        for i in range(n_samples):

            d_plus = np.dstack([d_plus_src] + [d_matrix_plus_tgt[:, i, :] for i in queries]).min(-1)
            d_minus = np.dstack([d_minus_src] + [d_matrix_minus_tgt[:, i, :] for i in queries]).max(-1)

            deltas = self.loss_fct(d_plus, d_minus)
            if self.verbose:
                print("Objective: %.3f"%deltas.mean())

            resize_d_plus = np.repeat(d_plus[:, np.newaxis, :], len(tgt_index), axis=1)
            resize_d_minus = np.repeat(d_minus[:, np.newaxis, :], len(tgt_index), axis=1)

            deltas_d_plus_minus = self.loss_fct(resize_d_plus, resize_d_minus)
            deltas_d_plus_matrix_minus_tgt = self.loss_fct(resize_d_plus, d_matrix_minus_tgt)
            deltas_d_minus_matrix_plus_tgt = self.loss_fct(d_matrix_plus_tgt, resize_d_minus)
            deltas_d_matrix_plus_minus_tgt = self.loss_fct(d_matrix_plus_tgt, d_matrix_minus_tgt)

            stacked_matrixes = np.dstack((deltas_d_plus_minus,
                            deltas_d_plus_matrix_minus_tgt,
                            deltas_d_minus_matrix_plus_tgt,
                            deltas_d_matrix_plus_minus_tgt))
            potential = stacked_matrixes.min(axis=-1).sum(0)

            queries.append(np.argmin(potential))

        d_plus = np.dstack([d_plus_src] + [d_matrix_plus_tgt[:, i, :] for i in queries]).min(-1)
        d_minus = np.dstack([d_minus_src] + [d_matrix_minus_tgt[:, i, :] for i in queries]).max(-1)

        deltas = self.loss_fct(d_plus, d_minus)
        if self.verbose:
            print("Final bjective: %.3f"%deltas.mean())
        
        self.indexes_ = np.array(queries, dtype=np.int32)
        self.n_samples_ = n_samples
        return self

    

class KMedoidsMiniBatch(BaseQueryRepresentative):
    
    def __init__(self, distance=manhattan_distances, **kwargs):
        self.distance = distance
        self.kwargs = kwargs
        super().__init__()
    
    
    def fit(self, X, y, src_index, tgt_index, n_samples, **fit_params):
        Xt = X[tgt_index]
        Xs = X[src_index]
        
        self.minikmeds = MiniBatchKMedoids(n_samples=n_samples, distance=self.distance, **self.kwargs)
        self.minikmeds.fit_nearest_neighbours(Xs, Xt)
        self.minikmeds.select_indexes = np.random.choice(len(Xt),
                                                self.minikmeds.batch_size_init,
                                                replace=False)
            
        self.minikmeds._initialization(Xt[self.minikmeds.select_indexes])
        
        self.indexes_ = np.array(self.minikmeds.queries, dtype=np.int32)
        self.X_ = Xt
        self.n_samples_ = n_samples
        return self
        
    def predict(self, n_samples=None):
        if n_samples is None:
            n_samples = self.n_samples_
        self.minikmeds.n_samples = n_samples
        self.minikmeds._kmedoids(self.X_, self.minikmeds.queries[:n_samples])
        
        return self.minikmeds.centers_index
    
    
class KMedoidsGreedy(BaseQueryRepresentative):

    def __init__(self, distance=manhattan_distances):
        self.distance = distance
        super().__init__()

    
    def fit(self, X, y, src_index, tgt_index, n_samples, **fit_params):
        queries = []

        distance_matrix_src = self.distance(X[tgt_index], X[src_index])
        deltas_src = np.min(distance_matrix_src, axis=1).ravel()
        distance_matrix_tgt = self.distance(X[tgt_index])

        for i in range(n_samples):
            deltas = np.min(np.stack(
                [deltas_src] + [distance_matrix_tgt[i] for i in queries],
                axis=1), axis=1)
            deltas_matrix = np.resize(deltas, distance_matrix_tgt.shape)
            diff_matrix = np.clip(deltas_matrix - distance_matrix_tgt, a_min=0., a_max=None)
            queries.append(diff_matrix.dot(np.ones((len(deltas_src), 1))).ravel().argmax())
            
        deltas = np.min(np.stack(
                [deltas_src] + [distance_matrix_tgt[i] for i in queries],
                axis=1), axis=1)
        
        self.indexes_ = np.array(queries, dtype=np.int32)
        self.n_samples_ = n_samples
        return self


class KMedoidsPAM(BaseQueryRepresentative):
    """
    """
    def __init__(self, distance=manhattan_distances, n_pam=2):
        self.distance = distance
        self.n_pam = n_pam
        super().__init__()

    
    def fit(self, X, y, src_index, tgt_index, n_samples, **fit_params):
        
        queries = []

        distance_matrix_src = self.distance(X[tgt_index], X[src_index])
        deltas_src = np.min(distance_matrix_src, axis=1).ravel()
        distance_matrix_tgt = self.distance(X[tgt_index])

        for i in range(n_samples):
            deltas = np.min(np.stack(
                [deltas_src] + [distance_matrix_tgt[i] for i in queries],
                axis=1), axis=1)
            deltas_matrix = np.resize(deltas, distance_matrix_tgt.shape)
            diff_matrix = np.clip(deltas_matrix - distance_matrix_tgt, a_min=0., a_max=None)
            queries.append(diff_matrix.dot(np.ones((len(deltas_src), 1))).ravel().argmax())
            

        self.deltas_src = deltas_src
        self.distance_matrix_tgt = distance_matrix_tgt
        
        self.indexes_ = np.array(queries, dtype=np.int32)
        self.X_ = X[tgt_index]
        self.n_samples_ = n_samples
        return self
    
    
    def predict(self, n_samples=None):
        assert self.X_ is not None
        assert self.indexes_ is not None
        
        if n_samples is None:
            n_samples = self.n_samples_
            
        assert n_samples <= len(self.indexes_)
        
        queries = list(self.indexes_[:n_samples])
        
        i = 0
        while i < self.n_pam * n_samples:
            index = i%len(queries)
            queries_whithout_index = copy.deepcopy(queries)
            queries_whithout_index.pop(index)
            deltas_whithout_index = np.min(np.stack(
            [self.deltas_src] + [self.distance_matrix_tgt[i] for i in queries_whithout_index],
            axis=1), axis=1)
            deltas_matrix = np.resize(deltas_whithout_index, self.distance_matrix_tgt.shape)
            diff_matrix = np.clip(deltas_matrix - self.distance_matrix_tgt, a_min=0., a_max=None)
            queries[index] = diff_matrix.dot(np.ones((len(self.deltas_src), 1))).ravel().argmax()
            i += 1
            
        deltas = np.min(np.stack(
                [self.deltas_src] + [self.distance_matrix_tgt[i] for i in queries],
                axis=1), axis=1)
            
        return np.array(queries)


class KMeansQuery(BaseQueryInformative):
    
    def fit(self, X, y, src_index, tgt_index, n_samples, **fit_params):
        self.information_ = np.ones(len(tgt_index))
        self.X_ = X[tgt_index]
        self.n_samples_ = n_samples
        return self


class RandomQuery(BaseQueryRepresentative):

    def fit(self, X, y, src_index, tgt_index, n_samples, **fit_params):
        queries = np.random.choice(len(tgt_index),
                                   n_samples,
                                   replace=False)
        self.indexes_ = np.array(queries, dtype=np.int32)
        self.n_samples_ = n_samples
        return self
    
    
class QueryByComitee(BaseQueryInformative):
    
    def __init__(self, get_model, n_models=10, minibatch=False, **kwargs):
        self.n_models = n_models
        self.get_model = get_model
        self.kwargs = kwargs
        super().__init__(minibatch=minibatch)
    
    def fit(self, X, y, src_index, tgt_index, n_samples, **fit_params):
        predictions = []
        for _ in range(self.n_models):
            src_index_boot = np.random.choice(src_index,
                                              size=len(src_index),
                                              replace=True)
            model = self.get_model(**self.kwargs, input_shape=X.shape[1:], output_shape=y.shape[1:])
            model.fit(X[src_index_boot], y[src_index_boot], **fit_params)
            y_pred = model.predict(X[tgt_index])
            predictions.append(y_pred)
        predictions = np.stack(predictions)
        confidence = np.std(predictions, axis=0).ravel()
        
        self.information_ = confidence
        self.X_ = X[tgt_index]
        self.n_samples_ = n_samples
        return self


class KcenterGreedy(BaseQueryRepresentative):
    
    def __init__(self, distance=manhattan_distances):
        self.distance = distance
        super().__init__()
    
    
    def fit(self, X, y, src_index, tgt_index, n_samples, **fit_params):
        queries = []        
        distance_matrix_src = self.distance(X[tgt_index], X[src_index])
        deltas = np.min(distance_matrix_src, axis=1).ravel()
        
        for i in range(n_samples):
            argmax = deltas.argmax()
            queries.append(argmax)
            deltas_tgt = self.distance(X[tgt_index],
                                             X[[tgt_index[argmax]]])
            deltas = np.min(np.concatenate((deltas.reshape(-1, 1),
                                            deltas_tgt.reshape(-1, 1)),
                                           axis=1), axis=1).ravel()

        self.indexes_ = np.array(queries, dtype=np.int32)
        self.n_samples_ = n_samples
        return self
    
    
class KcenterLargeScale(BaseQueryRepresentative):
    
    def __init__(self, distance=manhattan_distances):
        self.distance = distance
        super().__init__()
    
    def fit(self, X, y, src_index, tgt_index, n_samples, **fit_params):
        queries = []        
        
        kdtf = KDTreeForest(n_trees=50, distance=self.distance)
        kdtf.fit(X[src_index])
        deltas = kdtf.predict(X[tgt_index])
        
        for i in range(n_samples):
            argmax = deltas.argmax()
            queries.append(argmax)
            deltas_tgt = self.distance(X[tgt_index],
                                             X[[tgt_index[argmax]]])
            deltas = np.min(np.concatenate((deltas.reshape(-1, 1),
                                            deltas_tgt.reshape(-1, 1)),
                                           axis=1), axis=1).ravel()

        self.indexes_ = np.array(queries, dtype=np.int32)
        self.n_samples_ = n_samples
        return self


    
class DiversityLargeScale(BaseQueryRepresentative):
    
    def __init__(self, distance=manhattan_distances):
        self.distance = distance
        super().__init__()
    
    
    def fit(self, X, y, src_index, tgt_index, n_samples, **fit_params):
        queries = []
        
        kdtf = KDTreeForest(n_trees=50, distance=self.distance)
        kdtf.fit(X[src_index])
        deltas_src = kdtf.predict(X[tgt_index])
        deltas_list = deltas_src.reshape(-1, 1)
        
        queries = []
        for i in range(n_samples):
            deltas = np.mean(deltas_list, axis=1)
            argmax = deltas.argmax()
            queries.append(argmax)
            deltas_tgt = self.distance(X[tgt_index],
                                             X[[tgt_index[argmax]]])
            deltas_list = np.concatenate((deltas_list, deltas_tgt.reshape(-1, 1)), axis=1)
        
        self.indexes_ = np.array(queries, dtype=np.int32)
        self.n_samples_ = n_samples
        return self
    
    
    
class Diversity(BaseQueryRepresentative):
    
    def __init__(self, distance=manhattan_distances):
        self.distance = distance
        super().__init__()
    
    def fit(self, X, y, src_index, tgt_index, n_samples, **fit_params):
        queries = []
        
        distance_matrix_src = self.distance(X[tgt_index], X[src_index])
        deltas_src = np.mean(distance_matrix_src, axis=1).ravel()
        deltas_list = deltas_src.reshape(-1, 1)

        queries = []
        for i in range(n_samples):
            deltas = np.mean(deltas_list, axis=1)
            argmax = deltas.argmax()
            queries.append(argmax)
            deltas_tgt = self.distance(X[tgt_index],
                                             X[[tgt_index[argmax]]])
            deltas_list = np.concatenate((deltas_list, deltas_tgt.reshape(-1, 1)), axis=1)
        
        self.indexes_ = np.array(queries, dtype=np.int32)
        self.n_samples_ = n_samples
        return self
    
    
class BVSB(BaseQueryInformative):
    
    def __init__(self, get_model, minibatch=False, **kwargs):
        self.get_model = get_model
        self.kwargs = kwargs
        super().__init__(minibatch=minibatch)


    def fit(self, X, y, src_index, tgt_index, n_samples, **fit_params):
        
        queries = []
        
        model = self.get_model(**self.kwargs, input_shape=X.shape[1:], output_shape=y.shape[1:])
        model.fit(X[src_index], y[src_index], **fit_params)
        y_pred = model.predict(X[tgt_index])
        
        if len(y_pred.shape) == 1 or y_pred.shape[1] == 1:
            y_pred = y_pred.ravel()
            confidence = 0.5 - np.abs(0.5 - y_pred)
        else:
            sorted_pred = np.sort(y_pred, axis=1)
            confidence = 1 - (sorted_pred[:, -1] - sorted_pred[:, -2])
            confidence = confidence.ravel()
            
        self.information_ = confidence
        self.X_ = X[tgt_index]
        self.n_samples_ = n_samples
        return self
