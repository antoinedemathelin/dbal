"""
Run experiments
"""
import sys
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import manhattan_distances

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten, Reshape
from tensorflow.keras.constraints import MinMaxNorm
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

from adapt.instance_based import TrAdaBoostR2

from query_methods import *
from training_models import *
from utils import *


def run_superconductivity_k20():
    if not os.path.isfile("../datasets/superconductivity/train.csv"):
        os.makedirs("../datasets/superconductivity", exist_ok=True)
        download_superconductivity()
    
    def get_base_model(input_shape=(166,), output_shape=(1,), C=1):
        inputs = Input(shape=input_shape)
        modeled = Flatten()(inputs)
        modeled = Dense(100, activation='relu',
                             kernel_constraint=MinMaxNorm(0, C),
                             bias_constraint=MinMaxNorm(0, C))(modeled)
        modeled = Dense(100, activation='relu',
                             kernel_constraint=MinMaxNorm(0, C),
                             bias_constraint=MinMaxNorm(0, C))(modeled)
        modeled = Dense(np.prod(output_shape), activation=None,
                        kernel_constraint=MinMaxNorm(0, C),
                        bias_constraint=MinMaxNorm(0, C))(modeled)
        model = Model(inputs, modeled)
        model.compile(optimizer=Adam(0.001), loss='mean_squared_error')
        return model

    fit_params = dict(epochs=100, batch_size=128, verbose=0)
    n_queries_list = [20]
    max_queries = np.max(n_queries_list)
    source_list = [0, 1, 2, 3]
    target_list = [0, 1, 2, 3]
    training_list = [BalanceWeighting]
    n_source = None
    n_target = None

    logs = dict(random_state=[],
                source=[], target=[],
                n_queries=[],
                q_method=[],
                t_method=[],
                score=[])

    for random_state in range(8):
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        for source in source_list:
            for target in target_list:
                if source != target:
                    Xs, ys = load_superconductivity(source)
                    Xt, yt = load_superconductivity(target)

                    if n_source is not None:
                        index_ = np.random.choice(len(Xs), n_source, replace=False)
                        Xs = Xs[index_]
                        ys = ys[index_]
                    if n_target is not None:
                        index_ = np.random.choice(len(Xt), n_target, replace=False)
                        Xt = Xt[index_]
                        yt = yt[index_]

                    Xs, ys, Xt, yt, convert_y = preprocessing_superconductivity(Xs, ys, Xt, yt)

                    # Train ensemble of model for qbc
                    print("ensemble...")
                    models = [get_base_model() for _ in range(1)]
                    for mod in models:
                        index_boot = np.random.choice(len(Xs), size=len(Xs), replace=True)
                        mod.fit(Xs[index_boot], ys[index_boot], **fit_params)

                    Xs_emb = K.function([mod.layers[0].input],
                               [mod.get_layer(index=-2).output])(Xs)[0]
                    Xt_emb = K.function([mod.layers[0].input],
                               [mod.get_layer(index=-2).output])(Xt)[0]

                    print("random...")
                    random = RandomQuery()
                    random.fit(Xt_emb, Xs_emb, ys, max_queries)
                    print("kmeans...")
                    kmeans = KMeansQuery()
                    kmeans.fit(Xt_emb, Xs_emb, ys, max_queries)
                    print("diversity...")
                    divers = DiversityQuery()
                    divers.fit(Xt_emb, Xs_emb, ys, max_queries)
                    print("qbc...")
                    yt_preds = [mod.predict(Xt) for mod in models]
                    qbc = OrderedQuery()
                    qbc.fit(Xt_emb, Xs_emb, ys, max_queries, sample_weight=qbc_uncertainties(yt_preds))
                    print("kcenters...")
                    kcenters = KCentersQuery()
                    kcenters.fit(Xt_emb, Xs_emb, ys, max_queries)
                    print("kmedoids...")
                    kmedoids = KMedoidsQuery()
                    kmedoids.fit(Xt_emb, Xs_emb, ys, max_queries)

                    for n_queries in n_queries_list:
                        for q_method, q_name in zip([random, kmeans, divers, qbc, kcenters, kmedoids], 
                                                    ["random", "kmeans", "divers", "qbc", "kcenters", "kmedoids"]):
                            queries = q_method.predict(n_queries)
                            test_index = np.array(list(set(np.arange(len(Xt))) - set(queries)))
                            for t_method in training_list:
                                model = t_method(get_base_model)
                                model.fit(Xs, ys, Xt[queries], yt[queries], **fit_params);

                                y_pred = convert_y(model.predict(Xt[test_index]).ravel())
                                y_true = convert_y(yt[test_index])
                                score = mean_absolute_error(y_true, y_pred)
                                print("Source: %i, Target: %i, N: %i, Qmethod: %s, Tmethod: %s -> Score = %.3f"%
                                      (source, target, n_queries, q_name, t_method.__name__, score))
                                logs["random_state"].append(random_state)
                                logs["source"].append(source)
                                logs["target"].append(target)
                                logs["n_queries"].append(n_queries)
                                logs["q_method"].append(q_name)
                                logs["t_method"].append(t_method.__name__)
                                logs["score"].append(score)
    return logs
                                

def run_superconductivity_s2t3():
    if not os.path.isfile("../datasets/superconductivity/train.csv"):
        os.makedirs("../datasets/superconductivity", exist_ok=True)
        download_superconductivity()
    
    try:
        from adapt.instance_based import TrAdaBoostR2
        trada = True
    except:
        print("No module adapt found, please install adapt for TrAdaBoost experiments.")
        trada = False
    
    def get_base_model(input_shape=(166,), output_shape=(1,), C=1):
        inputs = Input(shape=input_shape)
        modeled = Flatten()(inputs)
        modeled = Dense(100, activation='relu',
                             kernel_constraint=MinMaxNorm(0, C),
                             bias_constraint=MinMaxNorm(0, C))(modeled)
        modeled = Dense(100, activation='relu',
                             kernel_constraint=MinMaxNorm(0, C),
                             bias_constraint=MinMaxNorm(0, C))(modeled)
        modeled = Dense(np.prod(output_shape), activation=None,
                        kernel_constraint=MinMaxNorm(0, C),
                        bias_constraint=MinMaxNorm(0, C))(modeled)
        model = Model(inputs, modeled)
        model.compile(optimizer=Adam(0.001), loss='mean_squared_error')
        return model

    fit_params = dict(epochs=100, batch_size=128, verbose=0)
    n_queries_list = [5, 10, 20, 30, 50, 75, 100, 150, 200, 250, 300]
    max_queries = np.max(n_queries_list)
    source_list = [2]
    target_list = [3]
    training_list = [UniformWeighting, BalanceWeighting]
    if trada:
         training_list.append(TrAdaBoostR2)
    n_source = None
    n_target = None

    logs = dict(random_state=[],
                source=[], target=[],
                n_queries=[],
                q_method=[],
                t_method=[],
                score=[])

    for random_state in range(8):
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        for source in source_list:
            for target in target_list:
                if source != target:
                    Xs, ys = load_superconductivity(source)
                    Xt, yt = load_superconductivity(target)

                    if n_source is not None:
                        index_ = np.random.choice(len(Xs), n_source, replace=False)
                        Xs = Xs[index_]
                        ys = ys[index_]
                    if n_target is not None:
                        index_ = np.random.choice(len(Xt), n_target, replace=False)
                        Xt = Xt[index_]
                        yt = yt[index_]

                    Xs, ys, Xt, yt, convert_y = preprocessing_superconductivity(Xs, ys, Xt, yt)
                    
                    # Train ensemble of model for qbc
                    print("ensemble...")
                    models = [get_base_model() for _ in range(1)]
                    for mod in models:
                        index_boot = np.random.choice(len(Xs), size=len(Xs), replace=True)
                        mod.fit(Xs[index_boot], ys[index_boot], **fit_params)

                    Xs_emb = K.function([mod.layers[0].input],
                               [mod.get_layer(index=-2).output])(Xs)[0]
                    Xt_emb = K.function([mod.layers[0].input],
                               [mod.get_layer(index=-2).output])(Xt)[0]

                    print("random...")
                    random = RandomQuery()
                    random.fit(Xt_emb, Xs_emb, ys, max_queries)
                    print("kmeans...")
                    kmeans = KMeansQuery()
                    kmeans.fit(Xt_emb, Xs_emb, ys, max_queries)
                    print("diversity...")
                    divers = DiversityQuery()
                    divers.fit(Xt_emb, Xs_emb, ys, max_queries)
                    print("qbc...")
                    yt_preds = [mod.predict(Xt) for mod in models]
                    qbc = OrderedQuery()
                    qbc.fit(Xt_emb, Xs_emb, ys, max_queries, sample_weight=qbc_uncertainties(yt_preds))
                    print("kcenters...")
                    kcenters = KCentersQuery()
                    kcenters.fit(Xt_emb, Xs_emb, ys, max_queries)
                    print("kmedoids...")
                    kmedoids = KMedoidsQuery()
                    kmedoids.fit(Xt_emb, Xs_emb, ys, max_queries)

                    for n_queries in n_queries_list:
                        for q_method, q_name in zip([random, kmeans, divers, qbc, kcenters, kmedoids], 
                                                    ["random", "kmeans", "divers", "qbc", "kcenters", "kmedoids"]):
                            queries = q_method.predict(n_queries)
                            test_index = np.array(list(set(np.arange(len(Xt))) - set(queries)))
                            for t_method in training_list:
                                if t_method.__name__ == "TrAdaBoostR2":
                                    model = t_method(get_base_model(input_shape=Xs.shape[1:]))
                                    model.fit(Xs=Xs,
                                              ys=ys,
                                              Xt=Xt[queries],
                                              yt=yt[queries],
                                              **fit_params)
                                else:
                                    model = t_method(get_base_model)
                                    model.fit(Xs, ys, Xt[queries], yt[queries], **fit_params)

                                y_pred = convert_y(model.predict(Xt[test_index]).ravel())
                                y_true = convert_y(yt[test_index])
                                score = mean_absolute_error(y_true, y_pred)
                                print("Source: %i, Target: %i, N: %i, Qmethod: %s, Tmethod: %s -> Score = %.3f"%
                                      (source, target, n_queries, q_name, t_method.__name__, score))
                                logs["random_state"].append(random_state)
                                logs["source"].append(source)
                                logs["target"].append(target)
                                logs["n_queries"].append(n_queries)
                                logs["q_method"].append(q_name)
                                logs["t_method"].append(t_method.__name__)
                                logs["score"].append(score)
    return logs



def run_office():
    def get_base_model(input_shape=(2048,), output_shape=(31,), activation="softmax", C=1):
        inputs = Input(input_shape)
        modeled = Dense(100, activation='relu',
                             kernel_constraint=MinMaxNorm(0, C),
                             bias_constraint=MinMaxNorm(0, C))(inputs)
        modeled = Dense(100, activation='relu',
                             kernel_constraint=MinMaxNorm(0, C),
                             bias_constraint=MinMaxNorm(0, C))(modeled)
        modeled = Dense(31, activation=activation,
                        kernel_constraint=MinMaxNorm(0, C),
                        bias_constraint=MinMaxNorm(0, C))(modeled)
        model = Model(inputs, modeled)
        model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy')
        return model

    fit_params = dict(epochs=60, batch_size=128, verbose=0)
    n_queries_list = [5, 10, 20, 30, 50, 75, 100, 150, 200, 250, 300]
    max_queries = np.max(n_queries_list)
    source = "amazon"
    target = "webcam"
    training_list = [BalanceWeighting]
    n_source = None
    n_target = None

    logs = dict(random_state=[],
                source=[], target=[],
                n_queries=[],
                q_method=[],
                t_method=[],
                score=[])

    for random_state in range(8):
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        Xs, ys = load_office(source)
        Xt, yt = load_office(target)

        if n_source is not None:
            index_ = np.random.choice(len(Xs), n_source, replace=False)
            Xs = Xs[index_]
            ys = ys[index_]
        if n_target is not None:
            index_ = np.random.choice(len(Xt), n_target, replace=False)
            Xt = Xt[index_]
            yt = yt[index_]

        Xs, ys, Xt, yt, convert_y = preprocessing_office(Xs, ys, Xt, yt)

        print("src_only...")
        src_only = get_base_model()
        src_only.fit(Xs, ys, **fit_params);

        Xs_emb = K.function([src_only.layers[0].input],
                   [src_only.get_layer(index=-2).output])(Xs)[0]
        Xt_emb = K.function([src_only.layers[0].input],
                   [src_only.get_layer(index=-2).output])(Xt)[0]
        
        yt_pred = src_only.predict(Xt)

        print("random...")
        random = RandomQuery()
        random.fit(Xt_emb, Xs_emb, ys, max_queries)
        print("kmeans...")
        kmeans = KMeansQuery()
        kmeans.fit(Xt_emb, Xs_emb, ys, max_queries)
        print("diversity...")
        divers = DiversityQuery()
        divers.fit(Xt_emb, Xs_emb, ys, max_queries)
        print("bvsb...")
        bvsb = OrderedQuery()
        bvsb.fit(Xt_emb, Xs_emb, ys, max_queries, sample_weight=bvsb_uncertainties(yt_pred))
        print("kcenters...")
        kcenters = KCentersQuery()
        kcenters.fit(Xt_emb, Xs_emb, ys, max_queries)
        print("clue...")
        uncertainties = yt_pred * np.log(yt_pred + 1e-6)
        uncertainties = -np.sum(uncertainties, axis=1).ravel()
        clue = KMeansQuery()
        clue.fit(Xt_emb, Xs_emb, ys, max_queries, sample_weight=uncertainties)
        print("kmedoids...")
        kmedoids = KMedoidsQuery()
        kmedoids.fit(Xt_emb, Xs_emb, ys, max_queries, sample_weight=bvsb_uncertainties(yt_pred))

        for n_queries in n_queries_list:
            for q_method, q_name in zip([random, kmeans, divers, bvsb, kcenters, clue, kmedoids],
                                        ["random", "kmeans", "divers", "bvsb", "kcenters", "clue", "kmedoids"]):
                queries = q_method.predict(n_queries)
                test_index = np.array(list(set(np.arange(len(Xt))) - set(queries)))
                for t_method in training_list:
                    model = t_method(get_base_model)
                    model.fit(Xs, ys, Xt[queries], yt[queries], **fit_params)

                    y_pred = convert_y(model.predict(Xt[test_index]))
                    y_true = convert_y(yt[test_index])
                    score = accuracy_score(y_true, y_pred)
                    print("Source: %s, Target: %s, N: %i, Qmethod: %s, Tmethod: %s -> Score = %.3f"%
                          (source, target, n_queries, q_name, t_method.__name__, score))
                    logs["random_state"].append(random_state)
                    logs["source"].append(source)
                    logs["target"].append(target)
                    logs["n_queries"].append(n_queries)
                    logs["q_method"].append(q_name)
                    logs["t_method"].append(t_method.__name__)
                    logs["score"].append(score)
    return logs



def run_digits():
    def get_base_model(input_shape=(768,), output_shape=10, activation="softmax", C=1):
        inputs = Input(input_shape)
        modeled = Dense(100, activation='relu',
                             kernel_constraint=MinMaxNorm(0, C),
                             bias_constraint=MinMaxNorm(0, C))(inputs)
        modeled = Dense(100, activation='relu',
                             kernel_constraint=MinMaxNorm(0, C),
                             bias_constraint=MinMaxNorm(0, C))(modeled)
        modeled = Dense(10, activation=activation,
                        kernel_constraint=MinMaxNorm(0, C),
                        bias_constraint=MinMaxNorm(0, C))(modeled)
        model = Model(inputs, modeled)
        model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy')
        return model

    fit_params = dict(epochs=30, batch_size=128, verbose=0)
    n_queries_list = [10, 20, 30, 50, 75, 100, 150, 200, 250, 300, 400, 500]
    max_queries = np.max(n_queries_list)
    source = "synth"
    target = "svhn"
    training_list = [BalanceWeighting]
    n_source = None
    n_target = None

    logs = dict(random_state=[],
                source=[], target=[],
                features=[],
                n_queries=[],
                q_method=[],
                t_method=[],
                score=[])

    for random_state in range(8):
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        for features in ["", "dann_"]:
            encoder = load_model("../datasets/models_digits/%sencoder.h5"%features)
            task = load_model("../datasets/models_digits/%stask.h5"%features)
            discriminator = load_model("../datasets/models_digits/%sdiscriminator.h5"%features)

            Xs, ys = load_digits(source)
            Xt, yt = load_digits(target)

            if n_source is not None:
                index_ = np.random.choice(len(Xs), n_source, replace=False)
                Xs = Xs[index_]
                ys = ys[index_]
            if n_target is not None:
                index_ = np.random.choice(len(Xt), n_target, replace=False)
                Xt = Xt[index_]
                yt = yt[index_]

            Xs = encoder.predict(Xs[:,:,:,np.newaxis])
            Xt = encoder.predict(Xt[:,:,:,np.newaxis])

            Xs, ys, Xt, yt, convert_y = preprocessing_digits(Xs, ys, Xt, yt)

            print("src_only...")
            src_only = get_base_model()
            src_only.fit(Xs, ys, **fit_params);

            Xs_emb = K.function([src_only.layers[0].input],
                       [src_only.get_layer(index=-2).output])(Xs)[0]
            Xt_emb = K.function([src_only.layers[0].input],
                       [src_only.get_layer(index=-2).output])(Xt)[0]
            
            yt_pred = src_only.predict(Xt)

            print("random...")
            random = RandomQuery()
            random.fit(Xt_emb, Xs_emb, ys, max_queries)
            print("kmeans...")
            kmeans = KMeansQuery(minibatch=True)
            kmeans.fit(Xt_emb, Xs_emb, ys, max_queries)
            print("diversity...")
            divers = DiversityQuery()
            sub_index = np.random.choice(len(Xs), 1000, replace=False)
            divers.fit(Xt_emb, Xs_emb[sub_index], ys[sub_index], max_queries)
            print("bvsb...")
            bvsb = OrderedQuery()
            bvsb.fit(Xt_emb, Xs_emb, ys, max_queries, sample_weight=bvsb_uncertainties(yt_pred))
            print("clue...")
            uncertainties = yt_pred * np.log(yt_pred + 1e-6)
            uncertainties = -np.sum(uncertainties, axis=1).ravel()
            clue = KMeansQuery(minibatch=True)
            clue.fit(Xt_emb, Xs_emb, ys, max_queries, sample_weight=uncertainties)
            print("kcenters...")
            kcenters = KCentersQuery(nn_algorithm="kdt-forest", n_trees=50)
            kcenters.fit(Xt_emb, Xs_emb, ys, max_queries)
            print("kmedoids...")
            kmedoids = KMedoidsAccelerated(nn_algorithm="kdt-forest", n_trees=50, batch_size_init=5000)
            kmedoids.fit(Xt_emb, Xs_emb, ys, max_queries, sample_weight=bvsb_uncertainties(yt_pred))
            print("aada...")
            y_disc = discriminator.predict(Xt).ravel()
            y_task = task.predict(Xt)
            aada = OrderedQuery()
            aada.fit(Xt_emb, Xs_emb, ys, max_queries, sample_weight=aada_uncertainties(y_task, y_disc))

            for n_queries in n_queries_list:
                for q_method, q_name in zip([random, kmeans, divers, bvsb, clue, kcenters, kmedoids, aada],
                                        ["random", "kmeans", "divers", "bvsb", "clue", "kcenters", "kmedoids", "aada"]):
                    queries = q_method.predict(n_queries)
                    test_index = np.array(list(set(np.arange(len(Xt))) - set(queries)))
                    for t_method in training_list:
                        model = t_method(get_base_model)
                        model.fit(Xs, ys, Xt[queries], yt[queries], **fit_params)

                        y_pred = convert_y(model.predict(Xt[test_index]))
                        y_true = convert_y(yt[test_index])
                        score = accuracy_score(y_true, y_pred)
                        print("Source: %s, Target: %s, Features: %s, N: %i, Qmethod: %s, Tmethod: %s -> Score = %.3f"%
                              (source, target, features, n_queries, q_name, t_method.__name__, score))
                        logs["random_state"].append(random_state)
                        logs["source"].append(source)
                        logs["features"].append(features)
                        logs["target"].append(target)
                        logs["n_queries"].append(n_queries)
                        logs["q_method"].append(q_name)
                        logs["t_method"].append(t_method.__name__)
                        logs["score"].append(score)
    return logs


if __name__ == "__main__":
    
    # Choose experiment between: "superconductivity_k20", "superconductivity_s2t3", "office", "digits"
    EXPERIMENT =  "superconductivity_k20"
    
    if EXPERIMENT == "superconductivity_k20":
        logs = run_superconductivity_k20()
    elif EXPERIMENT == "superconductivity_s2t3":
        logs = run_superconductivity_s2t3()
    elif EXPERIMENT == "office":
        logs = run_office()
    elif EXPERIMENT == "digits":
        logs = run_digits()
    else:
        raise ValueError("Choose experiment between: 'superconductivity_k20', 'superconductivity_s2t3', 'office', 'digits'")
            
    pd.DataFrame(logs).to_csv("%s.csv"%EXPERIMENT)