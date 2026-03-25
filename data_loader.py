import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from config import Config
import random

seed = Config.RANDOM_STATE
random.seed(seed)
np.random.seed(seed)


class Data():
    def __init__(self,
                 X: np.ndarray,
                 df: pd.DataFrame) -> None:
        self.X = X
        self.df = df
        X_DL = df[Config.TICKET_SUMMARY] + ' ' + df[Config.INTERACTION_CONTENT]
        X_DL = X_DL.to_numpy()
        y = df.y.to_numpy()
        y_series = pd.Series(y)
        good_y_value = y_series.value_counts()[y_series.value_counts() >= 3].index
        y_good = y[y_series.isin(good_y_value)]
        X_good = X[y_series.isin(good_y_value)]
        y_bad = y[y_series.isin(good_y_value) == False]
        X_bad = X[y_series.isin(good_y_value) == False]
        test_size = X.shape[0] * Config.TEST_SIZE / X_good.shape[0]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_good, y_good,     test_size=test_size, random_state=0)
        # X_train = np.concatenate((X_train, X_bad), axis=0)
        # y_train = np.concatenate((y_train, y_bad), axis=0)
        self.y = y_good
        self.classes = good_y_value
        self.embeddings = self.X

    def _attach_hierarchy_views(self):
        self.train_df = pd.DataFrame(self.train_df).reset_index(drop=True)
        self.test_df = pd.DataFrame(self.test_df).reset_index(drop=True)

        for frame in [self.train_df, self.test_df]:
            for col in ["y2", "y3", "y4"]:
                frame[col] = frame[col].fillna("").astype(str).str.strip()

        self.y2_train = self.train_df["y2"].to_numpy() if "y2" in self.train_df else None
        self.y2_test = self.test_df["y2"].to_numpy() if "y2" in self.test_df else None

        self.y3_train = self.train_df["y3"].to_numpy() if "y3" in self.train_df else None
        self.y3_test = self.test_df["y3"].to_numpy() if "y3" in self.test_df else None

        self.y4_train = self.train_df["y4"].to_numpy() if "y4" in self.train_df else None
        self.y4_test = self.test_df["y4"].to_numpy() if "y4" in self.test_df else None

        if "y2" in self.train_df and "y3" in self.train_df:
            y2_train = self.train_df["y2"].replace("", Config.MISSING_LABEL)
            y2_test = self.test_df["y2"].replace("", Config.MISSING_LABEL)
            y3_train = self.train_df["y3"].replace("", Config.MISSING_LABEL)
            y3_test = self.test_df["y3"].replace("", Config.MISSING_LABEL)
            self.y23_train = (y2_train + Config.CHAIN_SEPARATOR + y3_train).to_numpy()
            self.y23_test = (y2_test + Config.CHAIN_SEPARATOR + y3_test).to_numpy()

        if "y2" in self.train_df and "y3" in self.train_df and "y4" in self.train_df:
            y2_train = self.train_df["y2"].replace("", Config.MISSING_LABEL)
            y2_test = self.test_df["y2"].replace("", Config.MISSING_LABEL)
            y3_train = self.train_df["y3"].replace("", Config.MISSING_LABEL)
            y3_test = self.test_df["y3"].replace("", Config.MISSING_LABEL)
            y4_train = self.train_df["y4"].replace("", Config.MISSING_LABEL)
            y4_test = self.test_df["y4"].replace("", Config.MISSING_LABEL)
            self.y234_train = (
                y2_train + Config.CHAIN_SEPARATOR +
                y3_train + Config.CHAIN_SEPARATOR +
                y4_train
            ).to_numpy()

            self.y234_test = (
                y2_test + Config.CHAIN_SEPARATOR +
                y3_test + Config.CHAIN_SEPARATOR +
                y4_test
            ).to_numpy()

    def get_type(self):
        return self.y

    def get_X_train(self):
        return self.X_train

    def get_X_test(self):
        return self.X_test

    def get_type_y_train(self):
        return self.y_train

    def get_type_y_test(self):
        return self.y_test

    def get_embeddings(self):
        return self.embeddings

    def get_type_test_df(self):
        return self.test_df

    def get_X_DL_test(self):
        return self.X_DL_test

    def get_X_DL_train(self):
        return self.X_DL_train

    def get_level_target(self, level: str, split: str = "train"):
        mapping = {
            ("y2", "train"): self.y2_train,
            ("y2", "test"): self.y2_test,
            ("y3", "train"): self.y3_train,
            ("y3", "test"): self.y3_test,
            ("y4", "train"): self.y4_train,
            ("y4", "test"): self.y4_test,
            ("y23", "train"): getattr(self, "y23_train", None),
            ("y23", "test"): getattr(self, "y23_test", None),
            ("y234", "train"): getattr(self, "y234_train", None),
            ("y234", "test"): getattr(self, "y234_test", None),
        }
        return mapping[(level, split)]

    def get_level_data(self, level: str):
        y_train = self.get_level_target(level, "train")
        y_test = self.get_level_target(level, "test")

        if y_train is None or y_test is None:
            return None

        train_counts = pd.Series(y_train).value_counts()
        valid_classes = train_counts[train_counts >= Config.MIN_CLASS_COUNT].index

        keep_train = pd.Series(y_train).isin(valid_classes).to_numpy()
        keep_test = pd.Series(y_test).isin(valid_classes).to_numpy()

        X_train = self.X_train[keep_train]
        X_test = self.X_test[keep_test]
        y_train = y_train[keep_train]
        y_test = y_test[keep_test]

        if len(X_train) == 0 or len(X_test) == 0:
            return None

        if len(pd.unique(y_train)) < 2:
            return None

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }

    def get_branch_data(self, target_level: str, parent_filters: dict):
        train_mask = np.ones(len(self.train_df), dtype=bool)
        test_mask = np.ones(len(self.test_df), dtype=bool)

        for col, value in parent_filters.items():
            train_mask &= (self.train_df[col].to_numpy() == value)
            test_mask &= (self.test_df[col].to_numpy() == value)

        branch_train_df = self.train_df.loc[train_mask].reset_index(drop=True)
        branch_test_df = self.test_df.loc[test_mask].reset_index(drop=True)

        branch_X_train = self.X_train[train_mask]
        branch_X_test = self.X_test[test_mask]

        if branch_train_df.empty or branch_test_df.empty:
            return None

        y_train = branch_train_df[target_level].to_numpy()
        y_test = branch_test_df[target_level].to_numpy()

        train_counts = pd.Series(y_train).value_counts()
        valid_classes = train_counts[train_counts >= 3].index

        keep_train = pd.Series(y_train).isin(valid_classes).to_numpy()
        keep_test = pd.Series(y_test).isin(valid_classes).to_numpy()

        branch_train_df = branch_train_df.loc[keep_train].reset_index(drop=True)
        branch_test_df = branch_test_df.loc[keep_test].reset_index(drop=True)

        branch_X_train = branch_X_train[keep_train]
        branch_X_test = branch_X_test[keep_test]

        y_train = branch_train_df[target_level].to_numpy()
        y_test = branch_test_df[target_level].to_numpy()

        if len(pd.unique(y_train)) < 2:
            return None

        return {
            "X_train": branch_X_train,
            "X_test": branch_X_test,
            "y_train": y_train,
            "y_test": y_test,
            "train_df": branch_train_df,
            "test_df": branch_test_df,
        }
