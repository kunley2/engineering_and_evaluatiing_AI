import numpy as np
import pandas as pd
from config import  Config
import random

random.seed(Config.RANDOM_STATE)
np.random.seed(Config.RANDOM_STATE)

def get_tfidf_embd(df:pd.DataFrame):
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidfconverter = TfidfVectorizer(max_features=2000, min_df=4, max_df=0.90)
    data = df[Config.TICKET_SUMMARY] + ' ' + df[Config.INTERACTION_CONTENT]
    X = tfidfconverter.fit_transform(data).toarray()
    return X

def combine_embd(X1, X2):
    return np.concatenate((X1, X2), axis=1)

