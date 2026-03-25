import numpy as np
import pandas as pd
import random
from preprocessing import get_input_data, remove_duplication, noise_remover
from embeddings import get_tfidf_embd
from data_loader import Data
from chain_targets import build_chained_targets
from data_loader import ChainedData
from model import RandomForest
from model import AdaBoost
from model import HistGB
from model import SGD
from model import Voting
from model import RandomTreesEnsemble
from config import Config

class Pipeline:
    def __init__(self, mode: str = Config.DEFAULT_PIPELINE_MODE):
        seed = Config.RANDOM_STATE
        random.seed(seed)
        np.random.seed(seed)
        self.mode = mode
        self.model_classes = [
            ("RandomForest", RandomForest),
            ("Hist_GB", HistGB),
            ("SGD", SGD),
            ("AdaBoost", AdaBoost),
            ("Voting", Voting),
            ("RandomTreesEmbedding", RandomTreesEnsemble),
        ]

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
        if self.mode == 'chained':
            return ChainedData(X, df, Config.CHAIN_TARGET_COLUMNS)
        return Data(X, df)

    def run_single_model(self, model_name, data):
        print(model_name)
        model_class = dict(self.model_classes)[model_name]
        model = model_class(model_name, data.get_embeddings(), data.get_type())
        model.train(data)
        model.predict(data.get_X_test())
        model.print_results(data)

    def perform_modelling(self, data, name):
        self.model_predict(data, name)

    def run(self):
        df = self.load_data()
        df = self.preprocess_data(df)
        df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
        df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')
        grouped_df = df.groupby(Config.GROUPED)
        for name, group_df in grouped_df:
            print(name)
            if self.mode == 'chained':
                group_df = build_chained_targets(group_df)
            X, group_df = self.get_embeddings(group_df)
            data = self.get_data_object(X, group_df)
            if self.mode == 'chained' and not data.get_target_names():
                print(f"Skipping {name}: no valid chained targets found.")
                continue
            self.perform_modelling(data, group_df, name)
