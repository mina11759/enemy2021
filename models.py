from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Input, Reshape, Dropout, concatenate, Activation, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from metric import top1_acc, top2_acc, top3_acc, top4_acc, top5_acc


def fc_model(x_train, y_train,
             x_valid, y_valid,
             x_test, y_test,
             sent_num, num_label, model_name):

    EPOCHS = 100
    BATCH_SIZE = 32
    embed_dim = 512

    input_layer = Input(shape=(sent_num, embed_dim))
    dense_layer = Dense(1024, activation='relu')(input_layer)
    flatten = Flatten()(dense_layer)
    dropout_layer = Dropout(0.25)(flatten)
    output_layer = Dense(units=num_label, activation='softmax')(dropout_layer)

    model = Model(inputs=[input_layer], outputs=[output_layer])
    model.summary()
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=[top1_acc, top2_acc, top3_acc, top4_acc, top5_acc]
    )
    model.fit(x=x_train, y=y_train,
              validation_data=(x_valid, y_valid),
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              # validation_steps=EPOCHS
              )
    score = model.evaluate(x_test, y_test,
                           batch_size=BATCH_SIZE)
    print("loss : ", score[0])
    print("top1_acc_1 : ", score[1])
    print("top1_acc_2 : ", score[2])
    print("top1_acc_3 : ", score[3])
    print("top1_acc_4 : ", score[4])
    print("top1_acc_5 : ", score[5])

    model.save('result/{0}_sent.h5'.format(model_name))
    # model.save_weights('result/{0}_sent_weight.h5'.format(model_name))


def lstm_model():
    pass


def word_fc_model(x_train, y_train,
             x_valid, y_valid,
             x_test, y_test,
             word_num, num_label, model_name):

    EPOCHS = 100
    BATCH_SIZE = 32
    embed_dim = 300

    input_layer = Input(shape=(word_num, embed_dim))
    dense_layer = Dense(1024, activation='relu')(input_layer)
    flatten = Flatten()(dense_layer)
    dropout_layer = Dropout(0.25)(flatten)
    output_layer = Dense(units=num_label, activation='softmax')(dropout_layer)

    model = Model(inputs=[input_layer], outputs=[output_layer])
    model.summary()
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=[top1_acc, top2_acc, top3_acc, top4_acc, top5_acc]
    )
    model.fit(x=x_train, y=y_train,
              validation_data=(x_valid, y_valid),
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              # validation_steps=EPOCHS
              )
    score = model.evaluate(x_test, y_test,
                           batch_size=BATCH_SIZE)
    print("loss : ", score[0])
    print("top1_acc_1 : ", score[1])
    print("top1_acc_2 : ", score[2])
    print("top1_acc_3 : ", score[3])
    print("top1_acc_4 : ", score[4])
    print("top1_acc_5 : ", score[5])

    model.save('result/{0}_word.h5'.format(model_name))
    # model.save_weights('result/{0}_word_weight.h5'.format(model_name), save_format='tf')