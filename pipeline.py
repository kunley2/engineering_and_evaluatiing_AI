import random

import numpy as np
import pandas as pd

from chain_targets import build_chained_targets
from config import Config
from data_loader import ChainedData, Data
from embeddings import get_tfidf_embd
from model import AdaBoost
from model import ChainedModel
from model import HierarchyModel
from model import HistGB
from model import RandomForest
from model import RandomTreesEnsemble
from model import SGD
from model import Voting
from preprocessing import get_input_data, noise_remover, remove_duplication


class Pipeline:
    def __init__(self):
        seed = Config.RANDOM_STATE
        random.seed(seed)
        np.random.seed(seed)
        self.model_classes = None

    def load_data(self):
        df = get_input_data()
        return df

    def preprocess_data(self, df):
        df = remove_duplication(df)
        df = noise_remover(df)
        return df

    def get_embeddings(self, df: pd.DataFrame):
        X = get_tfidf_embd(df)
        return X, df

    def get_data_object(self, X: np.ndarray, df: pd.DataFrame):
        return Data(X, df)

    def run_single_model(self, model_name, model, data):
        print(model_name)
        model.train(data)
        model.predict(data.get_X_test())
        model.print_results(data)

    def run_target_models(self, data):
        for target_name in data.get_target_names():
            data.set_active_target(target_name)
            print(target_name)
            for model_name, model_class in self.model_classes:
                model = model_class(model_name, data.get_embeddings(), data.get_type())
                self.run_single_model(model_name, model, data)

    def model_predict(self, data, df, name):
        results = []

        print("RandomForest")
        model = RandomForest("RandomForest", data.get_embeddings(), data.get_type())
        model.train(data)
        model.predict(data.get_X_test())
        model.print_results(data)

        print("ChainedModel")
        model = ChainedModel("RandomForest", data.get_embeddings(), data.get_type())
        model.train(data)
        model.predict(data.get_X_test())
        model.print_results(data)

        print("HierarchyModel")
        model = HierarchyModel("RandomForest", data.get_embeddings(), data.get_type())
        model.train(data)
        model.predict(data.get_X_test())
        model.print_results(data)

        print("Hist_GB")
        model = HistGB("Hist_GB", data.get_embeddings(), data.get_type())
        model.train(data)
        model.predict(data.get_X_test())
        res = model.print_results(data)
        results.append(res)

        print("SGD")
        model = SGD("SGD", data.get_embeddings(), data.get_type())
        model.train(data)
        model.predict(data.get_X_test())
        model.print_results(data)

        print("AdaBoost")
        model = AdaBoost("AdaBoost", data.get_embeddings(), data.get_type())
        model.train(data)
        model.predict(data.get_X_test())
        model.print_results(data)

        print("Voting")
        model = Voting("Voting", data.get_embeddings(), data.get_type())
        model.train(data)
        model.predict(data.get_X_test())
        model.print_results(data)

        print("RandomTreesEmbedding")
        model = RandomTreesEnsemble("RandomTreesEmbedding", data.get_embeddings(), data.get_type())
        model.train(data)
        model.predict(data.get_X_test())
        model.print_results(data)

    def perform_modelling(self, data, df, name):
        self.model_predict(data, df, name)

    def run(self):
        df = self.load_data()
        df = self.preprocess_data(df)
        df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype("U")
        df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype("U")
        grouped_df = df.groupby(Config.GROUPED)
        for name, group_df in grouped_df:
            print(name)
            if self.model_classes is not None:
                group_df = build_chained_targets(group_df)
                X, group_df = self.get_embeddings(group_df)
                data = ChainedData(X, group_df, Config.CHAIN_TARGET_COLUMNS)
                if not data.get_target_names():
                    continue
                self.run_target_models(data)
                continue
            X, group_df = self.get_embeddings(group_df)
            data = self.get_data_object(X, group_df)
            self.perform_modelling(data, group_df, name)
