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
        self.embeddings = X

    def get_type(self):
        return  self.y
    def get_X_train(self):
        return  self.X_train
    def get_X_test(self):
        return  self.X_test
    def get_type_y_train(self):
        return  self.y_train
    def get_type_y_test(self):
        return  self.y_test
    def get_train_df(self):
        return  self.train_df
    def get_embeddings(self):
        return  self.embeddings
    def get_type_test_df(self):
        return  self.test_df
    def get_X_DL_test(self):
        return self.X_DL_test
    def get_X_DL_train(self):
        return self.X_DL_train


class ChainedData:
    def __init__(
        self,
        X: np.ndarray,
        df: pd.DataFrame,
        target_columns: dict[str, str],
    ) -> None:
        self.X = X
        self.df = df
        self.embeddings = X
        self.target_columns = target_columns
        self.target_splits: dict[str, dict[str, np.ndarray | pd.Index]] = {}
        self.active_target: str | None = None

        for target_name, column_name in target_columns.items():
            target_series = df[column_name].astype(str)
            valid_classes = (
                target_series.value_counts()[target_series.value_counts() >= Config.MIN_CLASS_COUNT].index
            )
            valid_mask = target_series.isin(valid_classes)

            if len(valid_classes) < 2 or valid_mask.sum() < 2:
                continue

            X_target = X[valid_mask.to_numpy()]
            y_target = target_series[valid_mask].to_numpy()
            test_size = min(
                X.shape[0] * Config.TEST_SIZE / X_target.shape[0],
                0.5,
            )
            split_kwargs = {
                "test_size": test_size,
                "random_state": Config.RANDOM_STATE,
            }
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_target,
                    y_target,
                    stratify=y_target,
                    **split_kwargs,
                )
            except ValueError:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_target,
                    y_target,
                    **split_kwargs,
                )
            self.target_splits[target_name] = {
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
                "y_all": y_target,
                "classes": valid_classes,
            }

        if self.target_splits:
            self.set_active_target(next(iter(self.target_splits)))

    def get_target_names(self) -> list[str]:
        return list(self.target_splits.keys())

    def set_active_target(self, target_name: str) -> None:
        split = self.target_splits[target_name]
        self.active_target = target_name
        self.X_train = split["X_train"]
        self.X_test = split["X_test"]
        self.y_train = split["y_train"]
        self.y_test = split["y_test"]
        self.y = split["y_all"]
        self.classes = split["classes"]

    def get_active_target(self) -> str | None:
        return self.active_target

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
