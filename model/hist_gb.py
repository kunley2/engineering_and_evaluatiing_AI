from model import BaseModel
from sklearn.ensemble import HistGradientBoostingClassifier
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from config import Config

class HistGB(BaseModel):
    def __init__(self,
                 model_name: str,
                 embeddings: np.ndarray,
                 y: np.ndarray) -> None:
        super(HistGB, self).__init__()
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y
        self.model = HistGradientBoostingClassifier(random_state=Config.RANDOM_STATE)
        self.predictions = None

    
    def train(self, data) -> None:
        self.model = self.model.fit(data.X_train, data.y_train)


    def predict(self, X_test: np.ndarray):
        predictions = self.model.predict(X_test)
        self.predictions = predictions  


    def print_results(self, data):
        print("classification report:", classification_report(data.y_test, self.predictions))
        print("-------------" * 5, "\n")
        print("confussion matrix:", confusion_matrix(data.y_test, self.predictions))
        print("-------------" * 5,"\n")
        print("accuracy score:", accuracy_score(data.y_test, self.predictions))

    def data_transform(self) -> None:
        ...