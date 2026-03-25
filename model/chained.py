import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from model import BaseModel
from model import RandomForest
from model import HistGB
from model import SGD
from model import AdaBoost
from model import Voting
from model import RandomTreesEnsemble


class LevelData:
    def __init__(self,
                 X_train: np.ndarray,
                 X_test: np.ndarray,
                 y_train: np.ndarray,
                 y_test: np.ndarray) -> None:
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test


class ChainedModel(BaseModel):
    LEVELS = [
        ("Type 2", "y2"),
        ("Type 2 + Type 3", "y23"),
        ("Type 2 + Type 3 + Type 4", "y234"),
    ]

    def __init__(self,
                 model_name: str,
                 embeddings: np.ndarray,
                 y: np.ndarray,
                 base_model_name: str = "RandomForest") -> None:
        super(ChainedModel, self).__init__()
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y
        self.base_model_name = base_model_name
        self.predictions = {}
        self.level_models = {}
        self.level_data = {}
        self.data_transform()

    def _build_base_model(self, y):
        if self.base_model_name == "RandomForest":
            return RandomForest(self.base_model_name, self.embeddings, y)
        if self.base_model_name == "Hist_GB":
            return HistGB(self.base_model_name, self.embeddings, y)
        if self.base_model_name == "SGD":
            return SGD(self.base_model_name, self.embeddings, y)
        if self.base_model_name == "AdaBoost":
            return AdaBoost(self.base_model_name, self.embeddings, y)
        if self.base_model_name == "Voting":
            return Voting(self.base_model_name, self.embeddings, y)
        if self.base_model_name == "RandomTreesEmbedding":
            return RandomTreesEnsemble(self.base_model_name, self.embeddings, y)
        raise ValueError(f"Unsupported base model: {self.base_model_name}")

    def train(self, data) -> None:
        for label, level in self.LEVELS:
            level_data = data.get_level_data(level)
            if level_data is None:
                continue

            model = self._build_base_model(level_data["y_train"])
            model.train(
                LevelData(
                    level_data["X_train"],
                    level_data["X_test"],
                    level_data["y_train"],
                    level_data["y_test"],
                )
            )
            self.level_models[label] = model
            self.level_data[label] = level_data

    def predict(self, X_test: np.ndarray):
        for label, model in self.level_models.items():
            level_data = self.level_data[label]
            model.predict(level_data["X_test"])
            self.predictions[label] = model.predictions

    def print_results(self, data):
        for label, _ in self.LEVELS:
            if label not in self.level_models:
                continue

            level_data = self.level_data[label]
            predictions = self.predictions[label]
            print(label.upper())
            print("classification report:", classification_report(level_data["y_test"], predictions, zero_division=0))
            print("-------------" * 5, "\n")
            print("confussion matrix:", confusion_matrix(level_data["y_test"], predictions))
            print("-------------" * 5, "\n")
            print("accuracy score:", accuracy_score(level_data["y_test"], predictions))
            print("-------------" * 5)

    def data_transform(self) -> None:
        ...
