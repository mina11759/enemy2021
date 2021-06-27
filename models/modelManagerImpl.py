from modelManager import ModelManager
from tensorflow.keras.models import Model, load_model
from trainOption import TrainOption
import keract
import numpy as np


class ModelManagerImpl(ModelManager):
    def __init__(self, model: Model, model_name, train_opt: TrainOption):
        super().__init__(model, model_name, train_opt)

    def get_gradient(self, x, y=None, layer_name=None):
        gradients = keract.get_gradients_of_activations(self.model, x, y, layer_names=layer_name)
        return np.squeeze(list(gradients.values())[0])

    def get_activations(self, x, layer_name=None):
        nodes = [layer.output for layer in self.model.layers
                 if layer.name == layer_name or layer_name is None]

        input_layer_outputs, layer_outputs = [], []
        [input_layer_outputs.append(node) if 'input_' in node.name else layer_outputs.append(node) for node in nodes]
        activations = keract.get_activations(self.model, x, nodes_to_evaluate=layer_outputs)
        return np.squeeze(list(activations.values()))

    def get_layer(self, index):
        return self.model.layers[index]

    def train_model(self, x_train, y_train, x_test, y_test):
        self.model.compile(
            optimizer=self.train_opt.opt,
            loss=self.train_opt.loss,
            metrics=self.train_opt.metrics
        )

        self.model.fit(x_train, y_train,
                       batch_size=self.train_opt.batch_size,
                       epochs=self.train_opt.epochs,
                       validation_data=(x_test, y_test),
                       shuffle=True)

        self.model.save('models/train_result/%s.h5' % self.model_name, save_format='tf')

    def test_model(self, text_x, test_y):
        pass

    def get_intermediate_output(self, layer, data):
        intermediate_layer_model = Model(inputs=self.model.input,
                                         outputs=self.model.get_layer(layer.name).output)
        return intermediate_layer_model.predict(np.expand_dims(data, axis=0))

    def load_model(self):
        self.model = load_model('models/train_result/%s.h5' % self.model_name)
        self.model.compile(
            optimizer=self.train_opt.opt,
            loss=self.train_opt.loss,
            metric=self.train_opt.metrics
        )
        self.model.summary()

    def get_prob(self, data):
        data = data[np.newaxis, :]
        prob = np.squeeze(self.model.predict(data))
        return prob

    def get_fc_layer(self):
        layers = []
        for idx, layer in enumerate(self.model.layers):
            if 'input' in layer.name \
                    or 'concatenate' in layer.name \
                    or idx == len(self.model.layers) - 1 \
                    or 'flatten' in layer.name:
                continue

            layer_type = self.__get_layer_type(layer.name)
            if layer_type == "dense":
                layers.append(layer)
        return layers

    def get_continuous_fc_layer(self):
        layers = []
        for idx in range(len(self.model.layers)):
            if 'input' in self.model.layers[idx].name \
                    or 'concatenate' in self.model.layers[idx].name \
                    or idx >= len(self.model.layers) - 2 \
                    or 'flatten' in self.model.layers[idx].name:
                continue

            layer_before_type = self.__get_layer_type(self.model.layers[idx].name)
            layer_after_type = self.__get_layer_type(self.model.layers[idx + 1].name)
            if layer_before_type == "dense" and layer_after_type == "dense":
                layers.append([self.model.layers[idx], self.model.layers[idx + 1]])
        return layers

    @staticmethod
    def __get_layer_type(layer_name):
        return layer_name.split('_')[0]