import numpy as np
import tensorflow as tf
from envs.bandit import Bandit
from modules import TwoLayerModule

class Solver(object):
    def __init__(self, n_bandits, player_pool_size, d_player_embed, n_epoch, **kwargs):
        self.env = Bandit(n_bandits)
        self.player_pool = tf.Variable(np.array([np.random.randn(d_player_embed) for _ in range(player_pool_size)]))
        self.player_ids = np.arange(player_pool_size)
        self.n_bandits = n_bandits
        self.player_pool_size = player_pool_size
        self.d_player_embed = d_player_embed
        self.n_epoch = n_epoch
        self._build_network(**kwargs)
        self.sess.run(tf.global_variables_initializer())

    def _build_network(self, n_steps, channels, module, learning_rate):
        channels = [self.d_player_embed] + channels
        doubles = [1] + [2] * (n_steps - 1)
        self.sess = tf.Session()
        self.modules = [module(channels[i] * doubles[i], channels[i + 1]) for i in range(n_steps)]
        self.idx = tf.placeholder(tf.int32, shape=(self.n_bandits,))
        self.features = tf.gather(self.player_pool, self.idx)
        self.labels = tf.placeholder(tf.int32, [self.n_bandits, self.n_bandits])
        nxt_in = self.features
        for i in range(n_steps):
            out = self.modules[i](nxt_in)
            if i != n_steps - 1:
                nxt_in = (tf.reduce_sum(out, axis=0) - out) / (self.n_bandits - 1)
                nxt_in = tf.concat([out, nxt_in], axis=1)
        self.actions = tf.layers.dense(out, self.n_bandits, kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.action_prob = tf.nn.softmax(self.actions)
        self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.labels, logits=self.actions)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_tf = self.optimizer.minimize(self.loss)

    def _build_network_alt(self, learning_rate, **kwargs):
        self.sess = tf.Session()
        self.idx = tf.placeholder(tf.int32, shape=(self.n_bandits,))
        self.features = tf.gather(self.player_pool, self.idx)
        self.labels = tf.placeholder(tf.int32, [self.n_bandits, self.n_bandits])
        out = TwoLayerModule(self.d_player_embed, self.n_bandits, hidden_dim=128)(self.features)
        self.action_prob = tf.nn.softmax(out)
        self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.labels, logits=out)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_tf = self.optimizer.minimize(self.loss)

    def train(self):
        running_average = 0
        for i in range(self.n_epoch):
            player_ids = np.random.choice(self.player_ids, self.n_bandits, replace=False)
            player_labels = self._get_label(player_ids)
            loss, prob, _ = self.sess.run([self.loss, self.action_prob, self.train_tf], feed_dict={self.idx : player_ids, self.labels : player_labels})
            score = self._get_score(prob)
            running_average = running_average * 0.99 + score * 0.01
            if i % 1000 == 0:
                print('loss {}'.format(loss))
                print(np.argmax(prob, axis=1))
                print('turn {} average score {}'.format(i, running_average))

    def _get_label(self, player_ids):
        labels = np.argsort(player_ids)
        labels_onehot = np.zeros((self.n_bandits, self.n_bandits))
        labels_onehot[np.arange(self.n_bandits), labels] = 1
        return labels_onehot

    def _get_score(self, prob):
        bandit_select = np.argmax(prob, axis=1)
        select = np.zeros((self.n_bandits, self.n_bandits), dtype=np.int32)
        select[np.arange(self.n_bandits), bandit_select] = 1
        for j in range(self.n_bandits):
               select[0] = np.bitwise_or(select[0], select[j])
        score = np.sum(select[0])
        return score