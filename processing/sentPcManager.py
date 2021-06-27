from processingManager import ProcessingManager
import json, re, string
from nltk.tokenize import sent_tokenize
from sklearn.preprocessing import OneHotEncoder
import tensorflow_hub as hub
import tensorflow as tf
from tqdm import tqdm
import numpy as np


# for Sentence Embedding using Universal Sentence Encoder
class SentPcManager(ProcessingManager):
    def __init__(self, data_name):
        self.data_name = data_name
        self.max_sent_num = 0
        self.min_sent_num = 0

    def clean_text(self, item):
        # 1. Remove \r
        # curr_title = item["issue_title"].replace("\r", " ")
        curr_description = item["description"].replace("\r", " ")
        curr_description = curr_description.replace("\n", " ")

        # 2. Remove URLs
        curr_description = re.sub(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", "",
            curr_description
        )

        # 3. Remove unnecessary element
        log_list = [
            "Stack trace:", "stack:", "Backtrace", "Trace:", "stack trace:",
            "calltrace:", "<!doctype", "<!DOCTYPE", "gdb info:",
            "<HTML>", "INFO", "chrome", "WARNING:", "Results:", ""
        ]

        for log in log_list:
            start_loc = curr_description.find(log)
            curr_description = curr_description[:start_loc]

        # 4. Remove hex code
        curr_description = re.sub(r"(\w+)0x\w", "", curr_description)

        # 5. change to lower case
        curr_description = curr_description.lower()

        # 6. Sentence Tokenize
        curr_description_tokens = sent_tokenize(curr_description)

        # 7. strip trailing puctuation marks
        curr_description_filter = [
            word.strip(string.punctuation) for word in curr_description_tokens
        ]

        # 8. Join the lists
        curr_data = [x for x in curr_description_filter if x]

        return curr_data

    def load_data(self, min_sent=1, max_sent=50):
        print("load and preprocess bug report dataset ..")
        with open(self.data_name, 'rt', encoding='UTF8') as file:
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
                assignee = item['owner']
                clean_data = self.clean_text(item)
                num_sentence = len(clean_data)

                if num_sentence == 1:
                    continue

                if num_sentence > max_sent or num_sentence < min_sent:
                    continue

            all_description.append(clean_data)
            all_assignee.append(assignee)

        max_sent_num = max(sentence_num_list)

        self.max_sent_num = max_sent_num

        return all_assignee, all_description

    def padding(self, description, max_sentence_num):
        print("progress sentence padding..")
        final_sent_list = []

        for sentences in tqdm(description):
            sent_length = len(sentences)
            if sent_length < max_sentence_num:
                temp_ones = np.ones((max_sentence_num - sent_length), dtype=str)
                np_sentences = np.array(sentences)
                padding_sentence = np.concatenate((np_sentences, temp_ones))
                final_sent_list.append(padding_sentence)

            else:
                final_sent_list.append(np.array(sentences))

        return final_sent_list # : list

    def embed_label(self, labels):
        print("progress label(assignee) embedding..")
        label_reshape = np.reshape(labels, (-1, 1))
        enc = OneHotEncoder()
        embedding = enc.fit_transform(label_reshape).toarray()
        return embedding

    def embed_feature(self, description):
        print("progress sentence embedding using universal sentence encoder (use)..")
        use_embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
        description_embedding_list = []
        try:
            with tf.device('/cpu:0'):
                for sentences in tqdm(description):
                    description_embedding = use_embed(sentences)
                    description_embedding_list.append(description_embedding)

        except RuntimeError as e:
            print(e)

        print()
        print("Sentence embedding end !")
        print()

        return description_embedding_list