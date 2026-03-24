import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from model import BaseModel
from model import RandomForest
from model import HistGB
from model import SGD
from model import AdaBoost
from model import Voting
from model import RandomTreesEnsemble


class BranchData:
    def __init__(self,
                 X_train: np.ndarray,
                 X_test: np.ndarray,
                 y_train: np.ndarray,
                 y_test: np.ndarray) -> None:
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test


class HierarchyModel(BaseModel):
    def __init__(self,
                 model_name: str,
                 embeddings: np.ndarray,
                 y: np.ndarray,
                 base_model_name: str = "RandomForest") -> None:
        super(HierarchyModel, self).__init__()
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y
        self.base_model_name = base_model_name
        self.predictions = None

        self.root_model = None
        self.level2_models = {}
        self.level3_models = {}

        self.pred_y2 = None
        self.pred_y3 = None
        self.pred_y4 = None
        self.data_transform()

    def _build_base_model(self, y):
        base_models = {
            "RandomForest": RandomForest,
            "Hist_GB": HistGB,
            "SGD": SGD,
            "AdaBoost": AdaBoost,
            "Voting": Voting,
            "RandomTreesEmbedding": RandomTreesEnsemble,
        }
        if self.base_model_name not in base_models:
            raise ValueError(f"Unsupported base model: {self.base_model_name}")
        return base_models[self.base_model_name](self.base_model_name, self.embeddings, y)

    def train(self, data) -> None:
        if data.y2_train is None or data.y3_train is None or data.y4_train is None:
            raise ValueError("HierarchyModel requires y2, y3 and y4 columns in the data object.")

        self.root_model = self._build_base_model(data.y2_train)
        self.root_model.train(
            BranchData(data.X_train, data.X_test, data.y2_train, data.y2_test)
        )

        for y2_value in pd.Series(data.y2_train).unique():
            branch = data.get_branch_data("y3", {"y2": y2_value})
            if branch is None:
                continue

            model = self._build_base_model(branch["y_train"])
            model.train(
                BranchData(
                    branch["X_train"],
                    branch["X_test"],
                    branch["y_train"],
                    branch["y_test"]
                )
            )
            self.level2_models[y2_value] = model

        for y2_value in pd.Series(data.y2_train).unique():
            branch_y3 = data.get_branch_data("y3", {"y2": y2_value})
            if branch_y3 is None:
                continue

            y3_values = pd.Series(branch_y3["train_df"]["y3"]).unique()

            for y3_value in y3_values:
                branch = data.get_branch_data("y4", {"y2": y2_value, "y3": y3_value})
                if branch is None:
                    continue

                model = self._build_base_model(branch["y_train"])
                model.train(
                    BranchData(
                        branch["X_train"],
                        branch["X_test"],
                        branch["y_train"],
                        branch["y_test"]
                    )
                )
                self.level3_models[(y2_value, y3_value)] = model

    def predict(self, X_test: np.ndarray):
        self.root_model.predict(X_test)
        self.pred_y2 = np.array(self.root_model.predictions, dtype=object)

        pred_y3 = np.array([None] * len(X_test), dtype=object)
        pred_y4 = np.array([None] * len(X_test), dtype=object)

        for y2_value, model in self.level2_models.items():
            idx = np.where(self.pred_y2 == y2_value)[0]
            if len(idx) == 0:
                continue

            model.predict(X_test[idx])
            pred_y3[idx] = model.predictions

        for i in range(len(X_test)):
            key = (self.pred_y2[i], pred_y3[i])
            if key not in self.level3_models:
                continue

            model = self.level3_models[key]
            model.predict(X_test[i:i+1])
            pred_y4[i] = model.predictions[0]

        self.pred_y3 = pred_y3
        self.pred_y4 = pred_y4

        self.predictions = np.array([
            f"{self.pred_y2[i]} || {self.pred_y3[i]} || {self.pred_y4[i]}"
            for i in range(len(X_test))
        ], dtype=object)

    def print_results(self, data):
        print("TYPE 2 RESULTS")
        print("classification report:", classification_report(data.y2_test, self.pred_y2, zero_division=0))
        print("-------------" * 5, "\n")
        print("confussion matrix:", confusion_matrix(data.y2_test, self.pred_y2))
        print("-------------" * 5, "\n")
        print("accuracy score:", accuracy_score(data.y2_test, self.pred_y2))
        print("-------------" * 5)

        valid_y3 = pd.Series(self.pred_y3).notna().to_numpy()
        if valid_y3.any():
            print("TYPE 3 RESULTS")
            print("classification report:", classification_report(
                data.y3_test[valid_y3],
                self.pred_y3[valid_y3],
                zero_division=0
            ))
            print("-------------" * 5, "\n")
            print("confussion matrix:", confusion_matrix(
                data.y3_test[valid_y3],
                self.pred_y3[valid_y3]
            ))
            print("-------------" * 5, "\n")
            print("accuracy score:", accuracy_score(
                data.y3_test[valid_y3],
                self.pred_y3[valid_y3]
            ))
            print("-------------" * 5)

        valid_y4 = pd.Series(self.pred_y4).notna().to_numpy()
        if valid_y4.any():
            print("TYPE 4 RESULTS")
            print("classification report:", classification_report(
                data.y4_test[valid_y4],
                self.pred_y4[valid_y4],
                zero_division=0
            ))
            print("-------------" * 5, "\n")
            print("confussion matrix:", confusion_matrix(
                data.y4_test[valid_y4],
                self.pred_y4[valid_y4]
            ))
            print("-------------" * 5, "\n")
            print("accuracy score:", accuracy_score(
                data.y4_test[valid_y4],
                self.pred_y4[valid_y4]
            ))
            print("-------------" * 5)

        hierarchical_path_score = np.mean(
            (self.pred_y2 == data.y2_test) &
            (self.pred_y3 == data.y3_test) &
            (self.pred_y4 == data.y4_test)
        )

        print("HIERARCHICAL PATH ACCURACY:", hierarchical_path_score)

    def data_transform(self) -> None:
        ...
