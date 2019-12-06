from keras.preprocessing.sequence import pad_sequences
import pandas as pd
from gensim.models import Word2Vec
from collections import Counter
import numpy as np
# 数据准备
def load_atec():
    """
    加载atec的数据集
    :return:
    """
    train = pd.read_table('atec/atec_nlp_sim_train.csv', header=None,encoding='utf-8-sig')  # (39346, 4)
    train.columns = ['line_num', 'q1', 'q2', 'label']

    train_add = pd.read_table('atec/atec_nlp_sim_train_add.csv', header=None,encoding='utf-8-sig')
    train_add.columns = ['line_num', 'q1', 'q2', 'label']
    print(len(train),len(train_add))
    train = pd.concat([train, train_add], axis=0)  # 按列 (102477, 4)
    return train[:1000]
def select_best_length(train,limit_ratio=0.95):
    """
    根据数据集的句子长度，选择最佳的样本max-length
    :param limit_ratio:句子长度覆盖度，默认覆盖95%以上的句子
    :return:
    """
    len_list = []
    max_length = 0
    cover_rate = 0.0
    for q1, q2 in zip(train['q1'], train['q2']):
        len_list.append(len(q1))
        len_list.append(len(q2))
    all_sent = len(len_list)
    sum_length = 0
    len_dict = Counter(len_list).most_common()
    for i in len_dict:
        sum_length += i[1] * i[0]
    average_length = sum_length / all_sent
    for i in len_dict:
        rate = i[1] / all_sent
        cover_rate += rate
        if cover_rate >= limit_ratio:
            max_length = i[0]
            break
    print('average_length:', average_length)
    print('max_length:', max_length)
    return max_length
def build_data(train):
    """
    构建数据集
    :return:
    """
    sample_x_left = train.q1.apply(lambda x: [char for char in x if char]).tolist()
    sample_x_right = train.q2.apply(lambda x: [char for char in x if char]).tolist()

    #  这里构建的 词表有问题
    #  vocabs = []
    # vocab.append("PAD")   句子长度不够，使用 PAD 充填
    # vocab.append("UNK")  不在此表中的词，设置为UNK

    vocabs = {'UNK'}
    for x_left, x_right in zip(sample_x_left, sample_x_right):
        for char in x_left + x_right:
            vocabs.add(char)
    sample_x = [sample_x_left, sample_x_right]
    sample_y = train.label.tolist()
    print(len(sample_x_left), len(sample_x_right))
    datas = [sample_x, sample_y]
    word_dict = {wd: index for index, wd in enumerate(list(vocabs))}
    vocab_path = 'vocab.txt'
    with open(vocab_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(list(vocabs)))
    return datas, word_dict
def train_w2v(datas):
    """
    训练词向量
    :return:
    """
    sents = datas[0][0] + datas[0][1]
    model = Word2Vec(sentences=sents, size=300, min_count=1)
    model.wv.save_word2vec_format('token_vec_300.bin', binary=False)
def load_pretrained_embedding():
    """
    加载预训练的词向量
    :return:
    """
    embedding_file = 'token_vec_300.bin'
    embeddings_dict = {}
    with open(embedding_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split(' ')
            if len(values) < 300:
                continue
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_dict[word] = coefs
    print('Found %s word vectors.' % len(embeddings_dict))
    return embeddings_dict
def build_embedding_matrix(word_dict, embedding_dict, VOCAB_SIZE, EMBEDDING_DIM):
    """
    加载词向量矩阵
    :return:
    """
    print(VOCAB_SIZE)
    embedding_matrix = np.zeros((VOCAB_SIZE + 1, EMBEDDING_DIM))  # 【716+1，300】
    print(embedding_matrix.shape)
    for word, i in word_dict.items():
        embedding_vector = embedding_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
def convert_data(datas, word_dict, MAX_LENGTH):
    """
    将数据转换成keras所能处理的格式
    :return:
    """
    sample_x = datas[0]
    sample_y = datas[1]
    sample_x_left = sample_x[0]
    sample_x_right = sample_x[1]
    left_x_train = [[word_dict[char] for char in data] for data in sample_x_left]
    right_x_train = [[word_dict[char] for char in data] for data in sample_x_right]
    y_train = [int(i) for i in sample_y]
    left_x_train = pad_sequences(left_x_train, MAX_LENGTH, padding='pre')
    right_x_train = pad_sequences(right_x_train, MAX_LENGTH, padding='pre')
    y_train = np.expand_dims(y_train, 2)
    return left_x_train, right_x_train, y_train
