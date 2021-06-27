import abc
from abc import ABCMeta


class ProcessingManager(metaclass=ABCMeta):
    def __init__(self, data_name):
        self.data_name = data_name

    @abc.abstractmethod
    def clean_text(self, item):
        pass

    @abc.abstractmethod
    def load_data(self):
        pass

    @abc.abstractmethod
    def padding(self, description, max_sentence_num):
        pass

    @abc.abstractmethod
    def embed_label(self, labels):
        pass

    @abc.abstractmethod
    def embed_feature(self, description):
        pass