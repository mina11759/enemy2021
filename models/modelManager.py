import abc
from abc import ABCMeta
from tensorflow.keras.models import Model
from trainOption import TrainOption


class ModelManager(metaclass=ABCMeta):
    def __init__(self, model: Model, model_name, train_opt: TrainOption):
        self.model = model
        self.model_name = model_name
        self.train_opt = train_opt

    @abc.abstractmethod
    def get_gradient(self, x, y=None, layer_name=None):
        pass

    @abc.abstractmethod
    def get_activations(self, x, layer_name=None):
        pass

    @abc.abstractmethod
    def get_layer(self, index):
        pass

    @abc.abstractmethod
    def train_model(self, x_train, y_train, x_test, y_test):
        pass

    @abc.abstractmethod
    def test_model(self, text_x, test_y):
        pass

    @abc.abstractmethod
    def get_intermediate_output(self, layer, data):
        pass

    @abc.abstractmethod
    def load_model(self):
        pass

    @abc.abstractmethod
    def get_prob(self, data):
        pass

    @abc.abstractmethod
    def get_fc_layer(self):
        pass

    @abc.abstractmethod
    def get_continuous_fc_layer(self):
        pass
