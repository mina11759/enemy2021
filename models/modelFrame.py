import abc
from abc import ABCMeta


class ModelFrame(metaclass=ABCMeta):
    @abc.abstractmethod
    def get_model(self):
        pass
