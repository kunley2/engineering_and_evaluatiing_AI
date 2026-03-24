from model import BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from config import Config

class Voting(BaseModel):
    def __init__(self,
                 model_name: str,
                 embeddings: np.ndarray,
                 y: np.ndarray) -> None:
        super(Voting, self).__init__()
        clf1 = LogisticRegression(multi_class='multinomial', random_state=1)
        clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
        clf3 = GaussianNB()
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y
        self.model = VotingClassifier(estimators=[
                 ('lr', clf1), ('rf', clf2), ('gnb', clf3)],
                   voting='hard'
        )
        self.predictions = None

    
    def train(self, data) -> None:
        self.model = self.model.fit(data.X_train, data.y_train)


    def predict(self, X_test: np.ndarray):
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
