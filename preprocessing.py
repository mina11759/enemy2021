import json
import re, string, nltk
from nltk.tokenize import sent_tokenize
from sklearn.preprocessing import OneHotEncoder
import tensorflow_hub as hub
import tensorflow as tf
from tqdm import tqdm
import numpy as np


def clean_word_list(item):
    # 1. Remove \r
    current_title = item["issue_title"].replace("\r", " ")
    current_desc = item["description"].replace("\r", " ")
    # 2. Remove URLs
    current_desc = re.sub(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        "",
        current_desc,
    )
    # 3. Remove Stack Trace
    start_loc = current_desc.find("Stack trace:")
    current_desc = current_desc[:start_loc]
    # 4. Remove hex code
    current_desc = re.sub(r"(\w+)0x\w+", "", current_desc)
    # current_title = re.sub(r"(\w+)0x\w+", "", current_title)
    # 5. Change to lower case
    current_desc = current_desc.lower()
    # current_title = current_title.lower()
    # 6. Tokenize
    current_desc_tokens = nltk.word_tokenize(current_desc)
    # current_title_tokens = nltk.word_tokenize(current_title)
    # 7. Strip trailing punctuation marks
    current_desc_filter = [
        word.strip(string.punctuation) for word in current_desc_tokens
    ]
    # current_title_filter = [
    #     word.strip(string.punctuation) for word in current_title_tokens
    # ]
    # 8. Join the lists
    current_data = current_desc_filter
    current_data = [x for x in current_data if x]  # list(filter(None, current_data))

    return current_data


def clean_sent_list(item):
    # 1. Remove \r
    # current_title = item["issue_title"].replace("\r", " ")
    current_desc = item["description"].replace("\r", " ")
    current_desc = current_desc.replace("\n", " ")
    # 2. Remove URLs
    current_desc = re.sub(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        "",
        current_desc,
    )
    # 3. Remove Stack Trace
    remove_log = [
        "Stack trace:", "stack:", "Backtrace ", "Trace:", "stack trace:",
        "calltrace:", "<!doctype", "<!DOCTYPE", "gdb info:",
        "(no debugging symbols found)", "<HTML>", "play ball, dammit.",
        "MochitestServer :", "chrome:", "Result:", "Reproduce:",
        "Error: package", "Bug observations:", "Reproducible:",
        "reproduce:", "here:", "WARNING:",
        # "Error page display:", "Error: package", ,
        # , "INFO",
        # "IP Config data:", "1. bookmark widget on visible bookmark bar ",
        #  "Additional info:",
        ":"
    ]

    for rm_log in remove_log:
        start_loc = current_desc.find(rm_log)
        current_desc = current_desc[:start_loc]

    # 4. Remove hex code
    current_desc = re.sub(r"(\w+)0x\w+", "", current_desc)

    # 5. Change to lower case
    current_desc = current_desc.lower()

    # 6. Tokenize
    current_desc_tokens = sent_tokenize(current_desc)

    # 7. Strip trailing punctuation marks
    current_desc_filter = [
        word.strip(string.punctuation) for word in current_desc_tokens
    ]

    # 8. Join the lists
    current_data = [x for x in current_desc_filter if x]  # list(filter(None, current_data))

    return current_data


def preprocess_dataset(dataset_name): # data extraction
    print("preprocess bug report dataset..")
    with open(dataset_name, 'rt', encoding='UTF8') as file:
        text = file.read()
        text = text.replace('" : NULL', '" : "NULL"')
        data = json.loads(text, strict=False)

    sentence_num_list = []
    all_assignee = []
    all_description = []
    for item in tqdm(data):
        if item['owner'] == 'nobody@mozilla.org':
            continue
        else:
            owner = item['owner']
            clean_data = clean_sent_list(item)
            sentence_num = len(clean_data)

            if sentence_num == 1:
                continue

            # if sentence_num > 50: # test
            #     continue

        all_description.append(clean_data)
        all_assignee.append(owner)
        sentence_num_list.append(sentence_num)

    max_sentence_num = max(sentence_num_list)
    # all_data[index][key] = value
    print("description size -> ", np.shape(all_description))
    print("assignee size -> ", np.shape(all_assignee))
    print("max_sentence_num -> ", max_sentence_num)

    print("description output type -> ", type(all_description))
    print("assignee output type -> ", type(all_assignee))
    print("max_sentence_num output type -> ", type(max_sentence_num))

    return all_assignee, all_description, max_sentence_num


def sentence_padding(description, max_sentence_num):
    print("progress sentence padding..")
    final_sentences = list()

    for sentences in tqdm(description):
        # print("sentence len : ", len(sentences))
        sent_len = len(sentences)
        if sent_len < max_sentence_num:
            temp_arr = np.ones((max_sentence_num - sent_len), dtype=str)
            sent = np.array(sentences)
            pad_sent = np.concatenate((sent, temp_arr))
            final_sentences.append(pad_sent)
        else:
            final_sentences.append(np.array(sentences))

    print("final_sentences output type -> ", type(final_sentences))
    print(final_sentences[4][:])

    # for i in final_sentences:
    #     print(len(i))

    print("final_sentences len -> ", len(final_sentences))
    # print("final_sentences shape -> ", np.shape(final_sentences)) # (?, ?)
    return final_sentences # : List


def label_embedding(labels):
    # print(labels)
    print("progress label(assignee) embedding..")
    print("input label shape : ", np.shape(labels))
    label_reshape = np.reshape(labels, (-1, 1))
    print("label_reshape : ", np.shape(label_reshape))
    # le = LabelEncoder()
    enc = OneHotEncoder()
    # embedding = np.array(le.fit_transform(label_reshape))
    # embedding = np.reshape(embedding, (-1, 1))
    embedding = enc.fit_transform(label_reshape).toarray()
    # print("label Encoding end")

    # print(embedding)
    print("label_embedding shape -> ", np.shape(embedding))
    print("label_embedding output type -> ", type(embedding))
    return embedding


def sentence_embedding(desc):
    print("progress sentence embedding using universal sentence encoder..")
    use_embedding = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
    description_embedding_list = []
    # print("Zebal confirm.... : ", np.shape(desc)) #
    try:
        with tf.device('/cpu:0'):
        # with tf.device('/device:GPU:0'):
            for sentences in tqdm(desc):
                description_embedding = use_embedding(sentences)
                # print(description_embedding)
                description_embedding_list.append(description_embedding)
    except RuntimeError as e:
        print(e)

    print()
    print("-- Universal Sentence Embedding End --")
    print()

    return description_embedding_list # tensor(?, 512)


# def word_preprocess_dataset(dataset_name):
#     print("Preprocessing {0} dataset: Start".format(dataset_name))
#     # The JSON file location containing the data for deep learning model training
#     # open_bugs_json = "./data/{0}/deep_data.json".format(dataset_name)
#     open_bugs_json = dataset_name
#
#     # Word2vec parameters
#     min_word_frequency_word2vec = 5
#     embed_size_word2vec = 200
#     context_window_word2vec = 5
#
#     # The bugs are loaded from the JSON file and the preprocessing is performed
#
#     with open(open_bugs_json) as data_file:
#         text = data_file.read()
#         # Fix json files for mozilla core and mozilla firefox
#         text = text.replace('" : NULL', '" : "NULL"')
#         data = json.loads(text, strict=False)
#
#     all_data = []
#     for item in data:
#         current_data = clean_word_list(item)
#         all_data.append(current_data)
#
#     print("Preprocessing {0} dataset: Word2Vec model".format(dataset_name))
#     # A vocabulary is constructed and the word2vec model is learned using the preprocessed data. The word2vec model provides a semantic word representation for every word in the vocabulary.
#     wordvec_model = Word2Vec(
#         all_data,
#         min_count=min_word_frequency_word2vec,
#         vector_size=embed_size_word2vec,
#         window=context_window_word2vec,
#     )
#
#     # Save word2vec model to use in the model again and again
#     wordvec_model.save("./data/{0}/word2vec.model".format(dataset_name))
#
#     # The data used for training and testing the classifier is loaded and the preprocessing is performed
#     for min_train_samples_per_class in [0, 5, 10, 20]:
#         print(
#             "Preprocessing {0} dataset: Classifier data {1}".format(
#                 dataset_name, min_train_samples_per_class
#             )
#         )
#         closed_bugs_json = "./data/{0}/classifier_data_{1}.json".format(
#             dataset_name, min_train_samples_per_class
#         )
#
#         with open(closed_bugs_json) as data_file:
#             text = data_file.read()
#             # Fix json files for mozilla core and mozilla firefox
#             text = text.replace('" : NULL', '" : "NULL"')
#             data = json.loads(text, strict=False)
#
#         all_data = []
#         all_owner = []
#         for item in data:
#             current_data = clean_word_list(item)
#             all_data.append(current_data)
#             all_owner.append(item["owner"])
#
#         # Save all data arrays to use in the model again and again
#         # np.save(
#         #     "./data/{0}/all_data_{1}.npy".format(
#         #         dataset_name, min_train_samples_per_class
#         #     ),
#         #     all_data,
#         # )
#         # np.save(
#         #     "./data/{0}/all_owner_{1}.npy".format(
#         #         dataset_name, min_train_samples_per_class
#         #     ),
#         #     all_owner,
#         # )
#
#         return all_data, all_owner