from keras.metrics import top_k_categorical_accuracy


def top1_acc(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=1)


def top2_acc(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)


def top3_acc(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)


def top4_acc(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=4)


def top5_acc(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)
