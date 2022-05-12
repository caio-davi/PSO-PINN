import tensorflow as tf
from numpy.random import uniform
from swarm import utils
from numpy import linspace


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
        cold_start=True,
        initialization_method=None,
        verbose=False,
        c_decrease=False,
        pre_trained_x=None,
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
            x_min (int, optional): The min value for the weights generation. Defaults to -1.
            x_max (int, optional): The max value for the weights generation. Defaults to 1.
            cold_start (bool, optional): Set the starting velocities to 0. Defaults to True.
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
        self.x_min = x_min
        self.x_max = x_max
        self.initialization_method = initialization_method
        self.x = (
            pre_trained_x if pre_trained_x is not None else self.build_swarm()
        )
        self.p = self.x
        self.loss_history = []
        self.f_p, self.grads = self.fitness_fn(self.p)
        self.g = self.p[tf.math.argmin(input=self.f_p).numpy()[0]]
        self.cold_start = cold_start
        self.v = self.start_velocities()
        self.verbose = verbose
        self.name = "PSO"
        self.c_decrease = c_decrease
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

    def update_pso_params(self):
        self.c1 = self.c1 - self.c1 / self.n_iter
        self.c2 = self.c2 - self.c2 / self.n_iter

    def start_velocities(self):
        """Start the velocities of each particle in the population (swarm). If 'self.cold_start' is 'TRUE', the swarm starts with velocity 0, which means stopped.

        Returns:
            tf.Tensor: The starting velocities.
        """
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
        f_x, self.grads = self.fitness_fn(self.x)
        self.loss_history.append(tf.reduce_mean(f_x).numpy())
        self.p = tf.where(f_x < self.f_p, self.x, self.p)
        self.f_p = tf.where(f_x < self.f_p, f_x, self.f_p)

    def update_g_best(self):
        """Update the *g-best* position."""
        self.g = self.p[tf.math.argmin(input=self.f_p).numpy()[0]]

    def step(self):
        """It runs ONE step on the particle swarm optimization."""
        r1, r2 = self.get_randoms()
        self.v = (
            self.b * self.v
            + self.c1 * r1 * (self.p - self.x)
            + self.c2 * r2 * (self.g - self.x)
        )
        self.x = self.x + self.v
        self.update_p_best()
        self.update_g_best()

    def train(self):
        """The particle swarm optimization. The PSO will optimize the weights according to the losses of the neural network, so this process is actually the neural network training."""
        for i in range(self.n_iter):
            self.step()
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
