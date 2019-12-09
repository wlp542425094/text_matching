import tensorflow as tf
from text_matching.abcnn import args

class Graph:

    def __init__(self, abcnn1=False, abcnn2=False):
        self.p = tf.placeholder(dtype=tf.int32, shape=(None, args.seq_length), name='p')
        self.h = tf.placeholder(dtype=tf.int32, shape=(None, args.seq_length), name='h')
        self.y = tf.placeholder(dtype=tf.int32, shape=None, name='y')
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='drop_rate')

        self.embedding = tf.get_variable(dtype=tf.float32, shape=(args.vocab_size, args.char_embedding_size),
                                         name='embedding')

        self.W0 = tf.get_variable(name="aW",     # (19,100)
                                  shape=(args.seq_length + 4, args.char_embedding_size),
                                  initializer=tf.contrib.layers.xavier_initializer(),
                                  regularizer=tf.contrib.layers.l2_regularizer(scale=0.0004))

        self.abcnn1 = abcnn1
        self.abcnn2 = abcnn2
        self.forward()

    def dropout(self, x):
        return tf.nn.dropout(x, keep_prob=self.keep_prob)

    def forward(self):
        p_embedding = tf.nn.embedding_lookup(self.embedding, self.p)
        h_embedding = tf.nn.embedding_lookup(self.embedding, self.h)

        #  扩展1个维度
        p_embedding = tf.expand_dims(p_embedding, axis=-1)
        h_embedding = tf.expand_dims(h_embedding, axis=-1)

        p_embedding = tf.pad(p_embedding, paddings=[[0, 0], [2, 2], [0, 0], [0, 0]])  #(?, 19, 100, 1)
        h_embedding = tf.pad(h_embedding, paddings=[[0, 0], [2, 2], [0, 0], [0, 0]])

        print("p_embedding:",p_embedding)

        if self.abcnn1:
            euclidean = tf.sqrt(tf.reduce_sum(
                tf.square(tf.transpose(p_embedding, perm=[0, 2, 1, 3]) - tf.transpose(h_embedding, perm=[0, 2, 3, 1])),
                axis=1) + 1e-6)                      # shape=(?, 19, 19)
            attention_matrix = 1 / (euclidean + 1)   # shape=(?, 19, 19)
            p_attention = tf.expand_dims(tf.einsum("ijk,kl->ijl", attention_matrix, self.W0), -1) #shape=(?, 19, 100, 1)
            h_attention = tf.expand_dims(
                tf.einsum("ijk,kl->ijl", tf.transpose(attention_matrix, perm=[0, 2, 1]), self.W0), -1)

            p_embedding = tf.concat([p_embedding, p_attention], axis=-1) # (?, 19, 100, 2),
            h_embedding = tf.concat([h_embedding, h_attention], axis=-1)
        p = tf.layers.conv2d(p_embedding,
                             filters=args.cnn1_filters, #  50
                             kernel_size=(args.filter_width, args.filter_height)) # shape=(?, 17, 1, 50)
        h = tf.layers.conv2d(h_embedding, # Tensor 输入
                             filters=args.cnn1_filters, # 50 卷积过滤器的数量
                             kernel_size=(args.filter_width, args.filter_height)) # (3,100)  卷积窗的高和宽

        p = self.dropout(p)
        h = self.dropout(h)

        #shape = (?, 17, 1, 50)
        if self.abcnn2:
            attention_pool_euclidean = tf.sqrt(
                tf.reduce_sum(tf.square(tf.transpose(p, perm=[0, 3, 1, 2]) - tf.transpose(h, perm=[0, 3, 2, 1])),
                              axis=1))   # 欧氏距离
            print("attention_pool_euclidean:",attention_pool_euclidean) #shape=(?, 17, 17)
            attention_pool_matrix = 1 / (attention_pool_euclidean + 1)  #shape=(?, 17, 17)
            print("attention_pool_matrix:",attention_pool_matrix)
            p_sum = tf.reduce_sum(attention_pool_matrix, axis=2, keep_dims=True) # shape=(?, 17, 1)
            h_sum = tf.reduce_sum(attention_pool_matrix, axis=1, keep_dims=True) # shape=(?, 1, 17)
            print("p_sum:",p_sum)
            print("h_sum:",h_sum)

            p = tf.reshape(p, shape=(-1, p.shape[1], p.shape[2] * p.shape[3])) # shape=(?, 17, 50)
            h = tf.reshape(h, shape=(-1, h.shape[1], h.shape[2] * h.shape[3])) # shape=(?, 17, 50)
            print("p:",p)
            print("h:",h)

            p = tf.multiply(p, p_sum)   # shape=(?, 17, 50)
            h = tf.multiply(h, tf.matrix_transpose(h_sum)) # shape=(?, 17, 50)
            print("p2:", p)
            print("h2:", h)
        else:
            p = tf.reshape(p, shape=(-1, p.shape[1], p.shape[2] * p.shape[3]))  #shape=(?, 17, 50)
            h = tf.reshape(h, shape=(-1, h.shape[1], h.shape[2] * h.shape[3]))  #shape=

        p = tf.expand_dims(p, axis=3) #shape=(?, 17, 50, 1)
        h = tf.expand_dims(h, axis=3)

        p = tf.layers.conv2d(p,
                             filters=args.cnn2_filters,   #50
                             kernel_size=(args.filter_width, args.cnn1_filters)) #(3,50)
        h = tf.layers.conv2d(h,
                             filters=args.cnn2_filters,
                             kernel_size=(args.filter_width, args.cnn1_filters))  # shape=(?, 15, 1, 50)

        p = self.dropout(p)
        h = self.dropout(h)

        p_all = tf.reduce_mean(p, axis=1)  #shape=(?, 1, 50)
        h_all = tf.reduce_mean(h, axis=1)

        x = tf.concat((p_all, h_all), axis=2) # shape=(?, 1, 100)
        x = tf.reshape(x, shape=(-1, x.shape[1] * x.shape[2])) # shape=(?, 100)
        out = tf.layers.dense(x, 50) # shape=(?, 50)
        logits = tf.layers.dense(out, 2) # shape=(?, 2)
        self.train(logits)

    def train(self, logits):
        y = tf.one_hot(self.y, args.class_size)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
        self.loss = tf.reduce_mean(loss)
        self.train_op = tf.train.AdamOptimizer(args.learning_rate).minimize(self.loss)
        prediction = tf.argmax(logits, axis=1)
        correct_prediction = tf.equal(tf.cast(prediction, tf.int32), self.y)
        self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))