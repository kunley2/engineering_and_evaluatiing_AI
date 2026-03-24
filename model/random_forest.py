import numpy as np
import pandas as pd
from model import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 
from config import Config

class RandomForest(BaseModel):
    def __init__(self,
                 model_name: str,
                 embeddings: np.ndarray,
                 y: np.ndarray) -> None:
        super(RandomForest, self).__init__()
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y
        self.model = RandomForestClassifier(n_estimators=100, random_state=Config.RANDOM_STATE)
        self.predictions = None
        self.data_transform()


    def train(self, data) -> None:
        self.model = self.model.fit(data.X_train, data.y_train) 


    def predict(self, X_test: pd.Series):
        predictions = self.model.predict(X_test)
        self.predictions = predictions

    def print_results(self, data):
        print("classification report:", classification_report(data.y_test, self.predictions, zero_division=0))
        print("-------------" * 5, "\n")
        print("confussion matrix:", confusion_matrix(data.y_test, self.predictions))
        print("-------------" * 5,"\n")
        print("accuracy score:", accuracy_score(data.y_test, self.predictions))


    def data_transform(self) -> None:
        ...
