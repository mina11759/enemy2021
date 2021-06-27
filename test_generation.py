import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow
import random
from metric import top1_acc, top2_acc, top3_acc, top4_acc, top5_acc
import numpy as np
from tqdm import tqdm

from sent_preprocessing import preprocess_dataset, sentence_embedding, sentence_padding, label_embedding


def sent_remove(desc): # 1개씩
    sent_num = len(desc)
    empty_sent_list = [desc[:idx] + ['<e_s>'] + desc[min(idx + 1, sent_num):] for idx in range(sent_num)]
    return empty_sent_list


def cal_sm_distance(orig_desc, adv_desc, use_embed):
    # use_embedding = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
    orig_use = [use_embed(orig_desc)]
    adv_use = [use_embed(adv_desc)]
    cos_sim = cosine_similarity(orig_use, adv_use)
    return cos_sim[0][0]


def test_sentence_padding(description, max_sentence_num):
    print("progress sentence padding..")
    final_sentences = list()

    for sentences in description:
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
    # print(final_sentences[4][:])

    # for i in final_sentences:
    #     print(len(i))

    print("final_sentences len -> ", len(final_sentences))
    # print("final_sentences shape -> ", np.shape(final_sentences)) # (?, ?)
    return final_sentences # : List


def test_sentence_embedding(desc, embed_model):
    # print("progress sentence embedding using universal sentence encoder..")
    # use_embedding = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
    description_embedding_list = []
    # print("Zebal confirm.... : ", np.shape(desc)) #
    try:
        with tensorflow.device('/cpu:0'):
        # with tf.device('/device:GPU:0'):
            for sentences in tqdm(desc):
                description_embedding = embed_model(sentences)
                # print(description_embedding)
                description_embedding_list.append(description_embedding)
    except RuntimeError as e:
        print(e)

    print()
    print("-- Universal Sentence Embedding End --")
    print()

    return description_embedding_list # tensor(?, 512)


def label_comp(orig_label_pr, pred_label_pr):
    pass


def cal_siv(orig_desc, adv_desc): # calculate sentence impact score(SIS)
    pass


def sorted_sent_impact_value(_list):
    return sorted(_list, reverse=True)


def test_generation(file_name, seed_num=1000):
    """
    :param file_name:
    :param pretrained_model:
    :param seed_num:
    :return: genetated test sample (feature_num, sentence_num, 512)
    """
    min_sent_num = 5
    max_sent_num = 35
    all_label, all_desc, max_sent_num = preprocess_dataset(file_name, min_sent_num, max_sent_num)
    instance_num = len(all_label)

    print("load pretrained model...")
    model_path = 'result/Google_Chromium.json_sent.h5'
    model = load_model(model_path)
    print("Complete load model !")

    temp_dict = dict()
    for idx in range(instance_num):
        temp_dict[all_label[idx]] = all_desc[idx]

    seed_dict = random.sample(temp_dict.items(), seed_num)
    seed_label = []
    raw_seed_desc = []

    for idx in range(seed_num):
        seed_label.append(seed_dict[idx][0])
        raw_seed_desc.append(seed_dict[idx][1])

    pad_seed_desc = sentence_padding(raw_seed_desc, max_sent_num)
    seed_desc = sentence_embedding(pad_seed_desc)
    final_seed_desc = []

    for sentences in seed_desc:
        sent = sentences.numpy()
        final_seed_desc.append(sent)

    final_seed_desc = np.asarray(final_seed_desc)

    seed_label = label_embedding(seed_label)
    seed_label = np.asarray(seed_label)

    # print("seed label complete")

    orig_predict_list = get_output_weight(final_seed_desc, model)
    adv_desc_pool = []

    # print(raw_seed_desc)
    print(np.shape(raw_seed_desc))
    print(raw_seed_desc[0]) # 1000
    print(np.shape(raw_seed_desc[0])) # (8,)
    print(np.shape(raw_seed_desc[1]))
    print(np.shape(raw_seed_desc[2]))

    embedding_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    print("Trying to mutate data..")
    for idx, desc in enumerate(tqdm(raw_seed_desc)): # desc.shape is [sentence_num,] expected sentence_num = 20
        mutated_desc_pool = sent_remove(desc)
        embed_m_desc_list = []
        for m_desc in mutated_desc_pool:
            print(m_desc)
            reshape_m_desc = np.reshape(m_desc, (1, np.shape(m_desc)[0]))
            pad_m_desc = test_sentence_padding(reshape_m_desc, max_sent_num)
            embed_m_desc = test_sentence_embedding(pad_m_desc, embedding_model)
            embed_m_desc_list.append(embed_m_desc)

        temp_instance_num = len(embed_m_desc_list)
        embed_m_desc_list = np.asarray(embed_m_desc_list) # expected shape (7, 34, 512)
        embed_m_desc_list = np.reshape(embed_m_desc_list, (temp_instance_num, 34, 512))
        new_predict_val_list = get_output_weight(embed_m_desc_list, model)

        orig_predict_val = orig_predict_list[idx]
        orig_label_idx = np.argmax(orig_predict_val)

        new_label_list = []
        sis_list = []
        for new_predict_val in new_predict_val_list:
            new_label_idx = np.argmax(new_predict_val)
            new_label_list.append(new_label_idx)

            # Calculate SIS
            if orig_label_idx == new_label_idx:
                f_od = orig_predict_val[orig_label_idx]
                f_md = new_predict_val[new_label_idx]
                sis = 1 - ((f_od - f_md) / max_sent_num)
                sis_list.append(sis)

            else:
                f_od_x = orig_predict_val[orig_label_idx]
                f_md_x = new_predict_val[orig_label_idx]
                f_md_y = new_predict_val[new_label_idx]
                f_od_y = orig_predict_val[new_label_idx]
                sis = 1 - (((f_od_x - f_md_x) + (f_md_y - f_od_y)) / max_sent_num)
                sis_list.append(sis)

        min_sis_idx = sis_list.index(min(sis_list))
        candidate_desc = mutated_desc_pool[min_sis_idx]
        candidate_label_idx = new_label_list[min_sis_idx]

        if orig_label_idx == candidate_label_idx:
            continue

        else:
            adv_desc_pool.append(candidate_desc)

        f = open('result/adv/{0}.txt'.format(str(idx)), 'w')
        for sentence in candidate_desc:
            f.write(sentence)
        f.close()

    print("Complete mutation !")
    print("[INFO]Num of mutated description : {}".format(len(adv_desc_pool)))


def cal_sent_impact_value(orig_idx, new_idx, orig_pd, new_pd, sent_num):
    if orig_idx == new_idx:
        siv = ((orig_pd[orig_idx] - new_pd[new_idx]) / sent_num)
    else:
        siv = (((orig_pd[orig_idx] - new_pd[new_idx]) + (new_pd[orig_idx] - new_pd[new_idx])) / sent_num)

    return siv


def load_model(model_path):
    dependencies = {
        'top1_acc': top1_acc,
        'top2_acc': top2_acc,
        'top3_acc': top3_acc,
        'top4_acc': top4_acc,
        'top5_acc': top5_acc,

    }
    return tensorflow.keras.models.load_model(model_path, custom_objects=dependencies) # h5


def get_output_weight(x, model):
    BATCH_SIZE = 32
    prediction = model.predict(x, batch_size=BATCH_SIZE, verbose=2)
    return prediction


if __name__ == '__main__':
    file_name = 'Google_Chromium.json'
    test_generation(file_name)
