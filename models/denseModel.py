from modelFrame import ModelFrame
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten
from tensorflow.keras.models import Model


class DenseModel(ModelFrame):
    def __init__(self, sent_num, embed_dim, label_num):
        super(DenseModel, self)

        self.sent_num = sent_num
        self.embed_dim = embed_dim
        self.label_num = label_num

    def get_model(self):
        input_layer = Input(shape=(self.sent_num, self.embed_dim))
        dense_layer = Dense(1024, activation='relu')(input_layer)
        flatten = Flatten()(dense_layer)
        dropout = Dropout(0.25)(flatten)
        output_layer = Dense(units=self.label_num, activation='softmax')(dropout)

        model = Model(inputs=[input_layer], outputs=[output_layer])
        return model