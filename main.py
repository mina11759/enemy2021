from sklearn.model_selection import train_test_split
from models import fc_model, word_fc_model
from sent_preprocessing import preprocess_dataset, sentence_embedding, sentence_padding, label_embedding
from word_preprocessing import word_preprocess_dataset, desc_word_embedding
import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', dest='data_name', default='', help='data project name')
    parser.add_argument('--embed', dest='embedding', default='', help='embedding method name')
    args = parser.parse_args()

    data = args.data_name
    embedding = args.embedding

    if embedding == 'sentence':
        assignee_label, description, max_sentence_num = preprocess_dataset(str(data))

        print("the number of assignee : ", len(assignee_label)) #
        print("the number of description : ", len(description))

        num_assignee = len(assignee_label) # label num
        num_description = len(description) # feature num

        val_size = 0.2
        range = int(num_description * val_size)

        # sentence padding
        description = sentence_padding(description, max_sentence_num)
        description = sentence_embedding(description)
        final_desc = []

        for sentences in description:
            sent = sentences.numpy()
            final_desc.append(sent)

        final_desc = np.asarray(final_desc)
        # vectorization phase
        label = label_embedding(assignee_label) # label embedding
        final_label = np.asarray(label)
        print("label complete")
        label_num = np.shape(label)[1]

        train_x, test_x, train_y, test_y = train_test_split(final_desc, final_label, test_size=0.2, shuffle=True)

        x_train = train_x[:range]
        x_val = train_x[range:]
        y_train = train_y[:range]
        y_val = train_y[range:]

        fc_model(x_train, y_train,
                 x_val, y_val,
                 test_x, test_y,
                 max_sentence_num, label_num, model_name=str(data))

    if embedding == 'word':

        assignee_label, description, max_word_num = word_preprocess_dataset(str(data))

        print("the number of assignee : ", len(assignee_label))  #
        print("the number of description : ", len(description))

        num_assignee = len(assignee_label)  # label num
        num_description = len(description)  # feature num

        # ---------------- test ----------------- #
        # test_size = 0.01
        # range = int(num_description * test_size)
        #
        # description = description[:range]
        # assignee_label = assignee_label[:range]
        #
        # num_assignee_test = len(assignee_label)  # label num test
        # num_description_test = len(description)  # feature num test
        # ---------------- test ----------------- #

        val_size = 0.2
        range = int(num_description * val_size)
        # range_test = int(num_description_test * val_size)

        description, noise_list = desc_word_embedding(description, max_word_num)

        label = np.delete(assignee_label, noise_list)
        label = label_embedding(label)  # label embedding
        print("label complete")
        print("final label shape : ", np.shape(label))
        label_num = np.shape(label)[1]
        print("label num : ", label_num)

        train_x, test_x, train_y, test_y = train_test_split(description, label, test_size=0.2)

        x_train = np.array(train_x[:range])
        x_val = np.array(train_x[range:])
        y_train = np.array(train_y[:range])
        y_val = np.array(train_y[range:])
        x_test = np.array(test_x)
        y_test = np.array(test_y)

        word_fc_model(x_train, y_train,
                 x_val, y_val,
                 x_test, y_test,
                 max_word_num, label_num, model_name=str(data))