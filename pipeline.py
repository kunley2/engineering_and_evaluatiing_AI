import numpy as np
import pandas as pd
import random
from preprocessing import get_input_data, remove_duplication, noise_remover
from embeddings import get_tfidf_embd
from data_loader import Data
from model import RandomForest
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

    def model_predict(self, data, df, name):
        results = []
        print("RandomForest")
        model = RandomForest("RandomForest", data.get_embeddings(), data.get_type())
        model.train(data)
        model.predict(data.X_test)
        model.print_results(data)

        print("Hist_GB")
        model = HistGB("Hist_GB", data.get_embeddings(), data.get_type())
        model.train(data)
        model.predict(data.X_test)
        res = model.print_results(data)
        results.append(res)

        print("SGD")
        model = SGD("SGD", data.get_embeddings(), data.get_type())
        model.train(data)
        model.predict(data.X_test)
        model.print_results(data)

        print("AdaBoost")
        model = AdaBoost("AdaBoost", data.get_embeddings(), data.get_type())
        model.train(data)
        model.predict(data.X_test)
        model.print_results(data)

        print("Voting")
        model = Voting("Voting", data.get_embeddings(), data.get_type())
        model.train(data)
        model.predict(data.X_test)
        model.print_results(data)

        print("RandomTreesEmbedding")
        model = RandomTreesEnsemble("RandomTreesEmbedding", data.get_embeddings(), data.get_type())
        model.train(data)
        model.predict(data.X_test)
        model.print_results(data)

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

