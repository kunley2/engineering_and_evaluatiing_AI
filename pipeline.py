import numpy as np
import pandas as pd
import random
from preprocessing import get_input_data, remove_duplication, noise_remover
from embeddings import get_tfidf_embd
from data_loader import Data
from model import RandomForest, HierarchyModel
from model import AdaBoost
from model import HistGB
from model import SGD
from model import Voting
from model import RandomTreesEnsemble
from config import Config

class Pipeline:
    def __init__(self):
        seed = Config.RANDOM_STATE
        random.seed(seed)
        np.random.seed(seed)
        self.base_model_classes = {
            "RandomForest": RandomForest,
            "Hist_GB": HistGB,
            "SGD": SGD,
            "AdaBoost": AdaBoost,
            "Voting": Voting,
            "RandomTreesEmbedding": RandomTreesEnsemble,
        }

    def load_data(self):
        # load the input data
        df = get_input_data()
        return df

    def preprocess_data(self, df):
        # De-duplicate input data
        df = remove_duplication(df)
        # remove noise in input data
        df = noise_remover(df)
        # translate data to english
        # df[Config.TICKET_SUMMARY] = translate_to_en(df[Config.TICKET_SUMMARY].tolist())
        return df

    def get_embeddings(self, df: pd.DataFrame):
        X = get_tfidf_embd(df)  # get tf-idf embeddings
        return X, df

    def get_data_object(self, X: np.ndarray, df: pd.DataFrame):
        return Data(X, df)

    def run_single_model(self, model_name, model, data):
        print(model_name)
        model.train(data)
        model.predict(data.X_test)
        model.print_results(data)

    def build_flat_model(self, model_name, data):
        return self.base_model_classes[model_name](model_name, data.get_embeddings(), data.get_type())

    def build_hierarchy_model(self, base_model_name, data):
        hierarchy_name = f"HierarchyModel[{base_model_name}]"
        return HierarchyModel(
            hierarchy_name,
            data.get_embeddings(),
            data.get_type(),
            base_model_name=base_model_name,
        )

    def model_predict(self, data, df, name):
        for model_name in Config.FLAT_MODELS:
            model = self.build_flat_model(model_name, data)
            self.run_single_model(model_name, model, data)

        for base_model_name in Config.HIERARCHY_BASE_MODELS:
            hierarchy_name = f"HierarchyModel[{base_model_name}]"
            model = self.build_hierarchy_model(base_model_name, data)
            self.run_single_model(hierarchy_name, model, data)

    def perform_modelling(self, data, df, name):
        self.model_predict(data, df, name)

    def run(self):
        df = self.load_data()
        df = self.preprocess_data(df)
        df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
        df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')
        grouped_df = df.groupby(Config.GROUPED)
        for name, group_df in grouped_df:
            print(name)
            X, group_df = self.get_embeddings(group_df)
            data = self.get_data_object(X, group_df)
            self.perform_modelling(data, group_df, name)
