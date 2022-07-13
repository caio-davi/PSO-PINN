import tensorflow as tf
import tensorflow_probability as tfp
from numpy import linspace
from swarm import utils


class gd:
    def __init__(
        self,
        loss_op,
        layer_sizes,
        n_iter=2000,
        pop_size=30,
        alpha=0.001,
        beta=0.9,
        x_min=-1,
        x_max=1,
        initialization_method="xavier",
        beta_1=0.99,
        beta_2=0.999,
        epsilon=1e-7,
        verbose=False,
    ):
        self.loss_op = loss_op
        self.layer_sizes = layer_sizes
        self.pop_size = pop_size
        self.n_iter = n_iter
        self.alpha = alpha
        self.beta = beta
        self.x_min = x_min
        self.x_max = x_max
        self.verbose = verbose
        self.initialization_method = initialization_method
        self.dim = utils.dimensions(layer_sizes)
        self.x = self.build_swarm()
        self.loss_history = []
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.m1 = tf.zeros([self.pop_size, self.dim])
        self.m2 = tf.zeros([self.pop_size, self.dim])
        self.alpha = alpha * tf.math.sqrt(1 - self.beta_2) / (1 - self.beta_1)
        self.name = "ADAM"
        self.verbose_milestone = linspace(0, n_iter, 11).astype(int)

    def build_swarm(self):
        """Creates the swarm following the selected initialization method.

        Args:
            initialization_method (str): Chooses how to initialize the Neural Net weights. Allowed to be one of "uniform", "xavier", or "log_logistic". Defaults to None, where it uses uniform initialization.

        Returns:
            tf.Tensor: The PSO swarm population. Each particle represents a neural network.
        """
        return utils.build_NN(
            self.pop_size, self.layer_sizes, self.initialization_method
        )

    def individual_fn(self, particle):
        w, b = utils.decode(particle, self.layer_sizes)
        loss, grad = self.loss_op(w, b)
        return loss, utils.flat_grad(grad)

    @tf.function
    def fitness_fn(self, x):
        f_x, grads = tf.vectorized_map(self.individual_fn, x)
        return f_x[:, None], grads

    def adam_update(self):
        self.m1 = self.beta_1 * self.m1 + (1 - self.beta_1) * self.grads
        self.m2 = self.beta_2 * self.m2 + (1 - self.beta_2) * tf.math.square(
            self.grads
        )
        return self.alpha * self.m1 / tf.math.sqrt(self.m2) + self.epsilon

    def first_momentum_update(self):
        self.m1 = self.beta * self.m1 + (1 - self.beta) * self.grads
        return self.alpha * self.m1

    def step(self):
        self.loss, self.grads = self.fitness_fn(self.x)
        self.x = self.x - self.adam_update()

    def train(self):
        for i in range(self.n_iter):
            self.step()
            self.loss_history.append(tf.reduce_mean(self.loss).numpy())
            if self.verbose and i in self.verbose_milestone:
                utils.progress(
                    (i / self.n_iter) * 100,
                    metric="loss",
                    metricValue=self.loss_history[-1],
                )
        if self.verbose:
            utils.progress(100)
            print()

    def get_best(self):
        return utils.decode(
            self.x[tf.math.argmin(input=self.loss).numpy()[0]],
            self.layer_sizes,
        )

    def get_swarm(self):
        return self.x
