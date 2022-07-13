import tensorflow as tf
from numpy import inf, linspace
from numpy.random import uniform
from swarm import utils


class pso:
    def __init__(
        self,
        loss_op,
        layer_sizes,
        n_iter=2000,
        pop_size=30,
        b=0.99,
        c1=8e-2,
        c2=5e-1,
        gd_alpha=5e-3,
        initialization_method=None,
        verbose=False,
        c_decrease=False,
        pre_trained_x=None,
        beta_1=0.99,
        beta_2=0.999,
        epsilon=1e-8,
    ):
        """The Particle Swarm Optimizer class. Specially built to deal with tensorflow neural networks.

        Args:
            loss_op (function): The fitness function for PSO.
            layer_sizes (list): The layers sizes of the neural net.
            n_iter (int, optional): Number of PSO iterations. Defaults to 2000.
            pop_size (int, optional): Population of the PSO swarm. Defaults to 30.
            b (float, optional): Inertia of the particles. Defaults to 0.9.
            c1 (float, optional): The *p-best* coeficient. Defaults to 0.8.
            c2 (float, optional): The *g-best* coeficient. Defaults to 0.5.
            gd_alpha (float, optional): Learning rate for gradient descent. Defaults to 0.00, so there wouldn't have any gradient-based optimization.
            initialization_method (_type_, optional): Chooses how to initialize the Neural Net weights. Allowed to be one of "uniform", "xavier", or "log_logistic". Defaults to None, where it uses uniform initialization.
            verbose (bool, optional): Shows info during the training . Defaults to False.
        """
        self.loss_op = loss_op
        self.layer_sizes = layer_sizes
        self.pop_size = pop_size
        self.dim = utils.dimensions(layer_sizes)
        self.n_iter = n_iter
        self.b = b
        self.c1 = c1
        self.c2 = c2
        self.initialization_method = initialization_method
        self.x = (
            pre_trained_x if pre_trained_x is not None else self.build_swarm()
        )
        self.f_x, self.grads = self.fitness_fn(self.x)
        self.p, self.f_p = self.x, self.f_x
        self.loss_history = []
        self.g = self.p[tf.math.argmin(input=self.f_p).numpy()[0]]
        self.v = self.start_velocities()
        self.verbose = verbose
        self.c_decrease = c_decrease
        self.verbose_milestone = linspace(0, n_iter, 11).astype(int)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = (epsilon,)
        self.m1 = tf.zeros([self.pop_size, self.dim])
        self.m2 = tf.zeros([self.pop_size, self.dim])
        self.gd_alpha = (
            gd_alpha * tf.math.sqrt(1 - self.beta_2) / (1 - self.beta_1)
        )
        self.name = "PSO" if self.gd_alpha == 0 else "PSO-GD"

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

    def update_pso_params(self):
        self.c1 = self.c1 - 2 * self.c1 / self.n_iter
        self.c2 = self.c2 + self.c2 / self.n_iter

    def start_velocities(self):
        """Start the velocities of each particle in the population (swarm) as `0`.

        Returns:
            tf.Tensor: The starting velocities.
        """
        return tf.zeros([self.pop_size, self.dim])

    def individual_fn(self, particle):
        """Auxiliary function to get the loss of each particle.

        Args:
            particle (tf.Tensor): One particle of the PSO swarm representing a full neural network.

        Returns:
            tuple: The loss value and the gradients.
        """
        w, b = utils.decode(particle, self.layer_sizes)
        loss, grad = self.loss_op(w, b)
        return loss, utils.flat_grad(grad)

    @tf.function
    def fitness_fn(self, x):
        """Fitness function for the whole swarm.

        Args:
            x (tf.Tensor): The swarm. All the particle's current positions. Which means the weights of all neural networks.

        Returns:
            tuple: the losses and gradients for all particles.
        """
        f_x, grads = tf.vectorized_map(self.individual_fn, x)
        return f_x[:, None], grads

    def get_randoms(self):
        """Generate random values to update the particles' positions.

        Returns:
            _type_: tf.Tensor
        """
        return uniform(0, 1, [2, self.dim])[:, None]

    def update_p_best(self):
        """Updates the *p-best* positions."""
        self.p = tf.where(self.f_x < self.f_p, self.x, self.p)
        self.f_p = tf.where(self.f_x < self.f_p, self.f_x, self.f_p)

    def update_g_best(self):
        """Update the *g-best* position."""
        self.g = self.p[tf.math.argmin(input=self.f_p).numpy()[0]]

    def gradient_descent(self):
        self.m1 = self.beta_1 * self.m1 + (1 - self.beta_1) * self.grads
        self.m2 = self.beta_2 * self.m2 + (1 - self.beta_2) * tf.math.square(
            self.grads
        )
        return self.gd_alpha * self.m1 / tf.math.sqrt(self.m2) + self.epsilon

    def step(self):
        """It runs ONE step on the particle swarm optimization."""
        r1, r2 = self.get_randoms()
        self.v = self.b * self.v + (1 - self.b) * (
            self.c1 * r1 * (self.p - self.x) + self.c2 * r2 * (self.g - self.x)
        )
        self.x = self.x + self.v - self.gradient_descent()
        self.f_x, self.grads = self.fitness_fn(self.x)
        self.update_p_best()
        self.update_g_best()

    def train(self):
        """The particle swarm optimization. The PSO will optimize the weights according to the losses of the neural network, so this process is actually the neural network training."""
        for i in range(self.n_iter):
            self.step()
            self.loss_history.append(tf.reduce_mean(self.f_x).numpy())
            if self.c_decrease:
                self.update_pso_params()
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
        """Return the *g-best*, the particle with best results after the training.

        Returns:
            tf.Tensor: the best particle of the swarm.
        """
        return utils.decode(self.g, self.layer_sizes)

    def get_swarm(self):
        """Return the swarm.

        Returns:
            tf.Tensor: The positions of each particle.
        """
        return self.x

    def set_n_iter(self, n_iter):
        """Set the number of iterations.
        Args:
            x (int): Number of iterations.
        """
        self.n_iter = n_iter
        self.verbose_milestone = linspace(0, n_iter, 11).astype(int)
