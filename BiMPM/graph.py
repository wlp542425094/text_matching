import tensorflow as tf
from text_matching.bimpm import args
import os

class Graph:
    def __init__(self):
        self.p = tf.placeholder(name='p', shape=(None, args.max_char_len), dtype=tf.int32)
        self.h = tf.placeholder(name='h', shape=(None, args.max_char_len), dtype=tf.int32)
        self.p_vec = tf.placeholder(name='p_word', shape=(None, args.max_word_len, args.word_embedding_len),
                                    dtype=tf.float32)
        self.h_vec = tf.placeholder(name='h_word', shape=(None, args.max_word_len, args.word_embedding_len),
                                    dtype=tf.float32)
        self.y = tf.placeholder(name='y', shape=(None,), dtype=tf.int32)
        self.keep_prob = tf.placeholder(name='keep_prob', dtype=tf.float32)

        self.embed = tf.get_variable(name='embed', shape=(args.char_vocab_len, args.char_embedding_len),
                                     dtype=tf.float32)

        for i in range(1, 9):
            setattr(self, f'w{i}', tf.get_variable(name=f'w{i}', shape=(args.num_perspective, args.char_hidden_size),
                                                   dtype=tf.float32))   # (12,512)

        self.forward()
        self.train()

    def LSTM(self, x):
        cell = tf.nn.rnn_cell.BasicLSTMCell(args.char_hidden_size)
        return tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

    def BiLSTM(self, x):
        fw_cell = tf.nn.rnn_cell.BasicLSTMCell(args.char_hidden_size)
        bw_cell = tf.nn.rnn_cell.BasicLSTMCell(args.char_hidden_size)

        return tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, x, dtype=tf.float32)

    def dropout(self, x):
        return tf.nn.dropout(x, self.keep_prob)

    # 多角度匹配的相似度函数
    def full_matching(self, metric, vec, w):
        w = tf.expand_dims(tf.expand_dims(w, 0), 2)
        metric = w * tf.stack([metric] * args.num_perspective, axis=1) # shape=(?, 12, 15, 512) tf.stack()是一个矩阵拼接函数

        vec = w * tf.stack([vec] * args.num_perspective, axis=1)  # 12   shape=(?, 12, 1, 512)

        m = tf.matmul(metric, tf.transpose(vec, [0, 1, 3, 2]))  # shape=(?, 12, 15, 1)

        n = tf.norm(metric, axis=3, keep_dims=True) * tf.norm(vec, axis=3, keep_dims=True) # shape=(?, 12, 15, 1)

        cosine = tf.transpose(tf.divide(m, n), [0, 2, 3, 1])

        return cosine

    def maxpool_full_matching(self, v1, v2, w):
        cosine = self.full_matching(v1, v2, w)
        max_value = tf.reduce_max(cosine, axis=2, keep_dims=True)
        return max_value

    def cosine(self, v1, v2):
        m = tf.matmul(v1, tf.transpose(v2, [0, 2, 1]))
        n = tf.norm(v1, axis=2, keep_dims=True) * tf.norm(v2, axis=2, keep_dims=True)
        # cosine = m / n
        cosine = tf.divide(m, n)
        return cosine

    def forward(self):
        # ----- Word Representation Layer -----
        # 字嵌入
        p_char_embedding = tf.nn.embedding_lookup(self.embed, self.p)
        h_char_embedding = tf.nn.embedding_lookup(self.embed, self.h)
        # 过一遍LSTM后作为字向量
        with tf.variable_scope("lstm_p", reuse=None):
            p_output, _ = self.LSTM(p_char_embedding) # shape=(?, 15, 512)
        with tf.variable_scope("lstm_h", reuse=None):
            h_output, _ = self.LSTM(h_char_embedding)
        print("p_output:",p_output)

        # 字向量和词向量拼接起来
        p_embedding = tf.concat((p_output, self.p_vec), axis=-1)   #(?,8,512)+(?,8,100) = (?,8,612)
        h_embedding = tf.concat((h_output, self.h_vec), axis=-1)   # shape=(?, 15, 612), dtype=float32)
        print("p_embedding:", p_embedding)

        p_embedding = self.dropout(p_embedding)
        h_embedding = self.dropout(h_embedding)

        # ----- Context Representation Layer -----
        # 论文中是取context，tf不会输出所有时刻的ctx，这里用输出值代替
        with tf.variable_scope("bilstm_p", reuse=tf.AUTO_REUSE):
            (p_fw, p_bw), _ = self.BiLSTM(p_embedding)
        with tf.variable_scope("bilstm_h", reuse=tf.AUTO_REUSE):
            (h_fw, h_bw), _ = self.BiLSTM(h_embedding)   # shape=(?, 15, 512)
        print("p_fw:", p_fw)
        p_fw = self.dropout(p_fw)
        p_bw = self.dropout(p_bw)
        h_fw = self.dropout(h_fw)
        h_bw = self.dropout(h_bw)
        print("h_fw[:, -1, :]:",h_fw[:, -1, :])   #shape=(?, 512)
        # ----- Matching Layer -----
        # 1、Full-Matching 多维度匹配，该层的目标是把每一个序列不同时刻的context与
        #   另一个序列的所有时刻的context做一个比较，并且考虑了两个方向
        p_full_fw = self.full_matching(p_fw, tf.expand_dims(h_fw[:, -1, :], 1), self.w1)  # shape=(?, 15, 1, 12)
        p_full_bw = self.full_matching(p_bw, tf.expand_dims(h_bw[:, 0, :], 1), self.w2)
        h_full_fw = self.full_matching(h_fw, tf.expand_dims(p_fw[:, -1, :], 1), self.w1)
        h_full_bw = self.full_matching(h_bw, tf.expand_dims(p_bw[:, 0, :], 1), self.w2)

        print("p_full_fw:", p_full_fw)

        # 2、Maxpooling-Matching
        max_fw = self.maxpool_full_matching(p_fw, h_fw, self.w3)  # shape=(?, 15, 1, 12)
        max_bw = self.maxpool_full_matching(p_bw, h_bw, self.w4)
        print("max_fw:", max_fw)

        # 3、Attentive-Matching
        # 计算权重即相似度
        fw_cos = self.cosine(p_fw, h_fw)    # shape=(?, 15, 15)
        bw_cos = self.cosine(p_bw, h_bw)
        print("fw_cos:", fw_cos)


        # 计算attentive vector
        p_att_fw = tf.matmul(fw_cos, p_fw)   # shape=(?, 15, 512)
        p_att_bw = tf.matmul(bw_cos, p_bw)
        h_att_fw = tf.matmul(fw_cos, h_fw)
        h_att_bw = tf.matmul(bw_cos, h_bw)
        print("p_att_fw:", p_att_fw)

        p_mean_fw = tf.divide(p_att_fw, tf.reduce_sum(fw_cos, axis=2, keep_dims=True)) # shape=(?, 15, 512)
        p_mean_bw = tf.divide(p_att_bw, tf.reduce_sum(bw_cos, axis=2, keep_dims=True))
        h_mean_fw = tf.divide(h_att_fw, tf.reduce_sum(fw_cos, axis=2, keep_dims=True))
        h_mean_bw = tf.divide(h_att_bw, tf.reduce_sum(fw_cos, axis=2, keep_dims=True))
        print("p_mean_fw:", p_mean_fw)

        p_att_mean_fw = self.full_matching(p_fw, p_mean_fw, self.w5)  # (?, 15, 15, 12)
        p_att_mean_bw = self.full_matching(p_bw, p_mean_bw, self.w6)
        h_att_mean_fw = self.full_matching(h_fw, h_mean_fw, self.w5)
        h_att_mean_bw = self.full_matching(h_bw, h_mean_bw, self.w6)
        print("p_att_mean_fw:", p_att_mean_fw)


        # 4、Max-Attentive-Matching
        p_max_fw = tf.reduce_max(p_att_fw, axis=2, keep_dims=True) # shape=(?, 15, 1)
        p_max_bw = tf.reduce_max(p_att_bw, axis=2, keep_dims=True)
        h_max_fw = tf.reduce_max(h_att_fw, axis=2, keep_dims=True)
        h_max_bw = tf.reduce_max(h_att_bw, axis=2, keep_dims=True)
        print("p_max_fw:",p_max_fw)

        p_att_max_fw = self.full_matching(p_fw, p_max_fw, self.w7) # shape=(?, 15, 15, 12)
        p_att_max_bw = self.full_matching(p_bw, p_max_bw, self.w8)
        h_att_max_fw = self.full_matching(h_fw, h_max_fw, self.w7)
        h_att_max_bw = self.full_matching(h_bw, h_max_bw, self.w8)
        print("p_max_fw:",p_att_max_fw)

        mv_p = tf.concat(
            (p_full_fw, max_fw, p_att_mean_fw, p_att_max_fw,
             p_full_bw, max_bw, p_att_mean_bw, p_att_max_bw),
            axis=2)  # shape=(?, 15, 64, 12)
        print("mv_p:",mv_p)

        mv_h = tf.concat(
            (h_full_fw, max_fw, h_att_mean_fw, h_att_max_fw,
             h_full_bw, max_bw, h_att_mean_bw, h_att_max_bw),
            axis=2)

        mv_p = self.dropout(mv_p)
        mv_h = self.dropout(mv_h)

        mv_p = tf.reshape(mv_p, [-1, mv_p.shape[1], mv_p.shape[2] * mv_p.shape[3]])  # shape=(?, 15, 768)
        mv_h = tf.reshape(mv_h, [-1, mv_h.shape[1], mv_h.shape[2] * mv_h.shape[3]])
        print("mv_p2",mv_p)

        # ----- Aggregation Layer -----
        with tf.variable_scope("bilstm_agg_p", reuse=tf.AUTO_REUSE):
            (p_f_last, p_b_last), _ = self.BiLSTM(mv_p)    # shape=(?, 15, 512)
        with tf.variable_scope("bilstm_agg_h", reuse=tf.AUTO_REUSE):
            (h_f_last, h_b_last), _ = self.BiLSTM(mv_h)
        print("p_f_last:",p_f_last)
        x = tf.concat((p_f_last, p_b_last, h_f_last, h_b_last), axis=2) # shape=(?, 15, 2048)
        print("x:",x)
        x = tf.reshape(x, shape=[-1, x.shape[1] * x.shape[2]])
        x = self.dropout(x)

        x = tf.layers.dense(x, 10000, activation='tanh')
        x = self.dropout(x)
        x = tf.layers.dense(x, 512)
        # x = self.dropout(x)
        self.tmp = x
        self.logits = tf.layers.dense(x, args.class_size)


    def train(self):
        y = tf.one_hot(self.y, args.class_size)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=self.logits)
        self.loss = tf.reduce_sum(loss)
        self.train_op = tf.train.AdamOptimizer(args.learning_rate).minimize(self.loss)
        self.predict = tf.argmax(self.logits, axis=1, name="predictions")
        correct_prediction = tf.equal(tf.cast(self.predict, tf.int32), self.y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")