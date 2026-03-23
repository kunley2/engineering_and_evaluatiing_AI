from abc import ABC, abstractmethod

class BaseModel(ABC):

    def __init__(self):
        ...


    @abstractmethod
    def train(self):
        """
        Train the model using ML Models for Multi-class and mult-label classification.
        :params: df is essential, others are model specific
        :return: classifier
        """
        ...


    @abstractmethod
    def predict(self):
        """
        To predict the label of the input data using the trained model.
        :params: df is essential, others are model specific 
        :return: predicted label
        """
        ... 

    @abstractmethod
    def print_results(self):
        """
        To print the evaluation results of the model.
        :params: df is essential, others are model specific 
        :return: None
        """
        return None

    @abstractmethod
    def data_transform(self):
        """
        To transform the input data into the format required by the model.
        :params: df is essential, others are model specific 
        :return: transformed data
        """
        return  
    
    # def build(self, values={}):
    #     values = values if isinstance(values, dict) else utils.string2any(values)
    #     self.__dict__.update(self.defaults)
    #     self.__dict__.update(values)
    #     return self
