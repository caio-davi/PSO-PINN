import tensorflow as tf
from swarm import utils

import sys


class fss:
    def __init__(
        self,
        loss_op,
        layer_sizes,
        iter=2000,
        pop_size=30,
        w_scale=100,
        stepInd=0.01,
        stepVol=0.01,
        x_min=-1,
        x_max=1,
        xavier_init=False,
        verbose=False,
        newOP=False,
    ):
        self.loss_op = loss_op
        self.layer_sizes = layer_sizes
        self.iter = iter
        self.pop_size = pop_size
        self.stepInd = stepInd
        self.stepIndDecay = self.stepInd / self.iter
        self.stepVol = stepVol
        self.stepVolDecay = self.stepVol / self.iter
        self.w_scale = w_scale
        self.x_min = x_min
        self.x_max = x_max
        self.xavier_init = xavier_init
        self.dim = utils.dimensions(layer_sizes)
        self.X, self.w = self.make_pop()
        self.f_X = self.f(self.X)
        self.verbose = verbose
        self.newOP = newOP
        self.loss_history = []

    def individual_fn(self, particle):
        w, b = utils.decode(particle, self.layer_sizes)
        loss, _ = self.loss_op(w, b)
        return -loss

    @tf.function
    def f(self, x):
        f_x = tf.vectorized_map(self.individual_fn, x)
        return f_x[:, None]

    def individual(self):
        step = (
            tf.random.uniform([self.pop_size, self.dim], self.x_min, self.x_max)
            * self.stepInd
        )
        x1 = tf.add(self.X, step)
        f_x1 = self.f(x1)
        x1 = tf.where(f_x1 > self.f_X, x1, self.X)
        f_x1 = tf.where(f_x1 > self.f_X, f_x1, self.f_X)
        step = tf.where(f_x1 > self.f_X, step, tf.zeros([self.pop_size, self.dim]))
        return x1, step, f_x1

    def instictive(self, x1, step, f_x1):
        if self.newOP:
            self.__instictive__(x1, step)
        else:
            self._instictive_(x1, step, f_x1)

    def _instictive_(self, x1, step, f_x1):
        sum = tf.add(f_x1, -self.f_X)
        self.X = tf.add(
            x1,
            utils.replacenan(
                tf.divide(tf.reduce_sum(step * sum, axis=0), tf.reduce_sum(sum))
            ),
        )
        self.f_X = self.f(self.X)
        self.loss_history.append(-tf.reduce_min(self.f_X).numpy())

    def __instictive__(self, x1, step):
        self.X = tf.add(x1, step)
        self.f_X = self.f(self.X)
        self.loss_history.append(-tf.reduce_min(self.f_X).numpy())

    def bari(self):
        den = tf.reduce_sum(self.w[:, None] * self.X, 0)
        return den / tf.reduce_sum(self.w)

    def feed(self, f_x1):
        df = self.f_X - f_x1
        df_mean = df / tf.reduce_max(df)
        w1 = tf.add(self.w, tf.squeeze(df_mean))
        return tf.clip_by_value(w1, 0, self.w_scale)

    def volitive(self, w1):
        rand = tf.scalar_mul(self.stepVol, tf.random.uniform([self.pop_size, 1], 0, 1))
        bari_vector = utils.replacenan(tf.add(self.X, -self.bari()))
        step = tf.multiply(rand, bari_vector)
        x_contract = tf.add(self.X, -step)
        x_dilate = tf.add(self.X, step)
        self.X = tf.where(w1[:, None] > self.w[:, None], x_contract, x_dilate)
        self.w = w1

    def make_pop(self):
        if self.xavier_init:
            X = utils.make_xavier_NN(self.pop_size,self.layer_sizes)
        else:
            X = tf.Variable(
                tf.random.uniform([self.pop_size, self.dim], self.x_min, self.x_max)
            )
        w = tf.Variable([self.w_scale / 2] * self.pop_size)
        return X, w

    def update_steps(self):
        self.stepInd = self.stepInd - self.stepIndDecay
        self.stepVol = self.stepVol - self.stepVolDecay

    def train(self):
        for i in range(self.iter):
            x1, step, f_x1 = self.individual()
            w1 = self.feed(f_x1)
            self.instictive(x1, step, f_x1)
            self.volitive(w1)
            self.update_steps()
            if self.verbose and i % (self.iter / 10) == 0:
                utils.progress((i / self.iter) * 100)
        if self.verbose:
            utils.progress(100)

    def get_best(self):
        return utils.decode(
            tf.unstack(self.X)[tf.math.argmax(tf.reshape(self.f_X, [self.pop_size]))],
            self.layer_sizes,
        )

    def get_swarm(self):
        return self.X
