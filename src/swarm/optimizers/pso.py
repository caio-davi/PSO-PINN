import tensorflow as tf
import tensorflow_probability as tfp

from numpy.random import uniform
from swarm_nn import utils

import sys


class pso:
    def __init__(
        self,
        loss_op,
        layer_sizes,
        n_iter=2000,
        pop_size=30,
        b=0.9,
        c1=0.8,
        c2=0.5,
        x_min=-1,
        x_max=1,
        gd_alpha=0.00,
        cold_start=True,
        initialization_method=None,
        verbose=False,
    ):
        self.loss_op = loss_op
        self.layer_sizes = layer_sizes
        self.pop_size = pop_size
        self.dim = utils.dimensions(layer_sizes)
        self.n_iter = n_iter
        self.b = b
        self.c1 = c1
        self.c2 = c2
        self.x_min = x_min
        self.x_max = x_max
        self.x = self.build_swarm(initialization_method)
        self.p = self.x
        self.loss_history = []
        self.f_p, self.grads = self.fitness_fn(self.p)
        self.g = self.p[tf.math.argmin(input=self.f_p).numpy()[0]]
        self.gd_alpha = gd_alpha
        self.cold_start = cold_start
        self.v = self.start_velocities()
        self.initialization_method = initialization_method
        self.verbose = verbose
        self.name = "PSO" if self.gd_alpha == 0 else "PSO-GD"

    def build_swarm(self, initialization_method):
        if initialization_method == "xavier":
            return utils.make_pop_NN(self.pop_size, self.layer_sizes)
        if initialization_method == "log_logistic":
            dist = tfp.distributions.LogLogistic(0, 0.1)
            return dist.sample([self.pop_size, self.dim])
        else:
            return tf.Variable(
                tf.random.uniform([self.pop_size, self.dim], self.x_min, self.x_max)
            )

    # def update_pso_params(self):
    #     # diff_c1_c2 = self.b1 * self.c1 * self.c2 / self.n_iter
    #     # diff_c2_alpha = self.b2 * self.c2
    #     # self.c1 = self.c1 - diff_c1_c2
    #     # self.c2 = self.c2 + diff_c1_c2 - diff_c2_alpha
    #     # self.gd_alpha = self.gd_alpha + diff_c2_alpha
    #     self.c1 = self.c1 - 2 * self.c1 / self.n_iter
    #     self.c2 = self.c2 - 2 * self.c2 / self.n_iter

    def start_velocities(self):
        if self.cold_start:
            return tf.zeros([self.pop_size, self.dim])
        else:
            return tf.Variable(
                tf.random.uniform(
                    [self.pop_size, self.dim],
                    -self.x_max - self.x_min,
                    self.x_max - self.x_min,
                )
            )

    def individual_fn(self, particle):
        w, b = utils.decode(particle, self.layer_sizes)
        loss, grad = self.loss_op(w, b)
        return loss, utils.flat_grad(grad)

    @tf.function
    def fitness_fn(self, x):
        f_x, grads = tf.vectorized_map(self.individual_fn, x)
        return f_x[:, None], grads

    def get_randoms(self):
        return uniform(0, 1, [2, self.dim])[:, None]

    def update_p_best(self):
        f_x, self.grads = self.fitness_fn(self.x)
        self.loss_history.append(tf.reduce_mean(f_x).numpy())
        self.p = tf.where(f_x < self.f_p, self.x, self.p)
        self.f_p = tf.where(f_x < self.f_p, f_x, self.f_p)

    def update_g_best(self):
        self.g = self.p[tf.math.argmin(input=self.f_p).numpy()[0]]

    def step(self):
        r1, r2 = self.get_randoms()
        self.v = (
            self.b * self.v
            + self.c1 * r1 * (self.p - self.x)
            + self.c2 * r2 * (self.g - self.x)
            - self.gd_alpha * self.grads
        )
        self.x = self.x + self.v
        self.update_p_best()
        self.update_g_best()

    def train(self):
        for i in range(self.n_iter):
            self.step()
            if self.verbose and i % (self.n_iter / 10) == 0:
                utils.progress(
                    (i / self.n_iter) * 100,
                    metric="loss",
                    metricValue=self.loss_history[-1],
                )
        if self.verbose:
            utils.progress(100)
            print()

    def get_best(self):
        return utils.decode(self.g, self.layer_sizes)

    def get_swarm(self):
        return self.x
