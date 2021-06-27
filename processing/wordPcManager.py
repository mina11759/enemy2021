from processingManager import ProcessingManager
import json
import re, string
from tqdm import tqdm
import numpy as np
import nltk
from nltk.corpus import stopwords
import gensim as gensim


class WordPcManager(ProcessingManager):
    def __init__(self, data_name):
        self.data_name = data_name
        self.min_word_num = 0
        self.max_word_num = 0

    def clean_text(self, item):
        # 1.Remove \r
        # current_title = item["issue_title"].replace("\r", " ")
        current_desc = item["description"].replace("\r", " ")
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
            "Error page display:", "Error: package", "WARNING:", "Results:",
            "MochitestServer :", "INFO", "chrome:", "Reproduce:", "Reproducible:",
            "IP Config data:", "1. bookmark widget on visible bookmark bar ",
            "Bug observations:", "Additional info:", "reproduce:", "here:",
            "Result:", ":", "license 'UNKNOWN'", "grep", "$",
            "[  FAILED  ] WorkerTest.FLAKY_SharedWorkerFastConstructor", "91668",
        ]

        for rm_log in remove_log:
            start_loc = current_desc.find(rm_log)
            # current_desc.strip(' '+current_desc+' ')
            current_desc = current_desc[:start_loc]

        # 4. Remove hex code
        current_desc = re.sub(r"(\w+)0x\w+", "", current_desc)

        # 5. Change to lower case
        current_desc = current_desc.lower()

        # 6. Tokenize
        current_desc_tokens = nltk.word_tokenize(current_desc)
        # 7. Stopword
        stop_words = set(stopwords.words('english'))
        current_desc_tokens_stopwords = []
        for word in current_desc_tokens:
            if word not in stop_words:
                current_desc_tokens_stopwords.append(word)

        # 7. Strip trailing punctuation marks
        current_desc_filter = [
            word.strip(string.punctuation) for word in current_desc_tokens_stopwords
        ]

        # 8. Join the lists
        current_data = current_desc_filter
        current_data = [x for x in current_data if x]  # list(filter(None, current_data))

        return current_data

    def load_data(self):
        print("preprocess bug report dataset..")
        with open(self.data_name, 'rt', encoding='UTF8') as file:
            text = file.read()
            text = text.replace('" : NULL', '" : "NULL"')
            data = json.loads(text, strict=False)

        word_num_list = []
        all_assignee = []
        all_description = []

        for item in tqdm(data):
            if item['owner'] == 'nobody@mozilla.org':
                continue
            else:
                owner = item['owner']
                clean_data = self.clean_text(item)
                word_num = len(clean_data)

                if word_num == 0:
                    continue

                if word_num > 250 or word_num < 15:  # test
                    # print(clean_data)
                    continue

            all_description.append(clean_data)
            all_assignee.append(owner)
            word_num_list.append(word_num)

        max_word_num = max(word_num_list)
        min_word_num = min(word_num_list)
        # all_data[index][key] = value
        print("description size -> ", np.shape(all_description))
        print("assignee size -> ", np.shape(all_assignee))
        print("max_word_num -> {0}".format(max_word_num))
        print("min_word_num -> {0}".format(min_word_num))

        self.min_word_num = min_word_num
        self.max_word_num = max_word_num

        # print("description output type -> ", type(all_description))
        # print("assignee output type -> ", type(all_assignee))
        # print("max_word_num output type -> ", type(max_word_num))

        return all_assignee, all_description

    def padding(self):
        pass

    def embed_label(self):
        pass

    def embed_feature(self, description, max_word_num):
        """
            :param desc: np.array((#feature_num, word_num))
            :return: ((#feature, word_num, wordvec_dim=300))
            """
        print("progress word embedding using word2vec model..")
        print("load word2vec model..")

        word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',
                                                                         binary=True)
        print("word2vec load complete")
        desc_word_list = []

        print("INFO: Word Vectorizing Start")
        for words in tqdm(description):  # words are tokenizing
            one_desc_word_list = []
            for one_word in words:
                if one_word in word2vec_model:
                    word_vec = word2vec_model[one_word]
                    one_desc_word_list.append(word_vec)
                else:
                    continue

            desc_word_list.append(one_desc_word_list)  # <-- exist empty list

        noise_idx = []
        # ---- test ---- #
        for idx, d in enumerate(desc_word_list):
            if np.shape(d) == (0,):
                noise_idx.append(idx)
                # print(idx, d)

        desc_word_list = np.delete(desc_word_list, noise_idx)

        for i, des in enumerate(desc_word_list):
            if np.shape(des) == (0,):
                print(i, des)

        print()
        print("complete word2vec")
        print()
        print("word padding ..")
        # print("INFO : desc_word_list's shape -> {0}".format(np.shape(desc_word_list))) # (138298,)
        # print("INFO : desc_word_list's shape[0] -> {0}".format(np.shape(desc_word_list[0]))) # (15, 300)

        # noise = 0

        final_word_embedding = []
        for w_vec in tqdm(desc_word_list):  # w_vec : word vector  # memory kill
            num_word = np.shape(w_vec)[0]
            if max_word_num > num_word:
                temp_vec = np.zeros(((max_word_num - num_word), 300))
                pad_vec = np.concatenate([w_vec, temp_vec])
                final_word_embedding.append(pad_vec)

            else:
                final_word_embedding.append(w_vec)

        # print("INFO : desc_word_list's shape -> {0}".format(np.shape(final_word_embedding))) # (138298,)
        # print("INFO : desc_word_list's shape[0] -> {0}".format(np.shape(final_word_embedding[0]))) # (15, 300)

        print()
        print("-- Word Embedding End --")
        print()

        # final_word_embedding = np.array(final_word_embedding)

        # test print
        # print("np.shape(final_word_embedding) : ", np.shape(final_word_embedding))

        return final_word_embedding, noise_idx  # (#feature num, #word num, 300)