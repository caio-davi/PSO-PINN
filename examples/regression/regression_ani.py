import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math

from swarm.optimizers.pso import pso

from swarm import utils


np.random.seed(1234)
tf.random.set_seed(1234)

# Parameters
layer_sizes = [1] + 3 * [5] + [1]
pop_size = 100
n_iter = 20
stepInd = 0.01
stepVol = 0.01
w_scale = 100
x_min = -1
x_max = 1
sample_size = 265
noise = 0.0


def objective(x, noise=0):
    return (x ** 3 - tf.sin(2 * math.pi * x)) + tf.random.normal(
        x.shape, 0, noise, tf.float32
    )


def get_loss(X, y):
    def _loss(w, b):
        with tf.GradientTape() as tape:
            tape.watch(w)
            tape.watch(b)
            pred = utils.multilayer_perceptron(w, b, X)
            loss = tf.reduce_mean((y - pred) ** 2)
        trainable_variables = w + b
        grads = tape.gradient(loss, trainable_variables)
        return loss, grads

    return _loss


X = tf.reshape(
    tf.Variable(np.linspace(x_min, x_max, sample_size), dtype="float32"),
    [sample_size, 1],
)
y = objective(X, noise)


opt_pso = pso(
    get_loss(X, y),
    layer_sizes,
    n_iter,
    pop_size,
    0.9,
    0.8,
    0.5,
    x_min=x_min,
    x_max=x_max,
    gd_alpha=0,
)

opt_pso_gd = pso(
    get_loss(X, y),
    layer_sizes,
    n_iter,
    pop_size,
    0.9,
    0.8,
    0.5,
    x_min=x_min,
    x_max=x_max,
    gd_alpha=1e-2,
)

opt_pso_gd2 = pso(
    get_loss(X, y),
    layer_sizes,
    n_iter,
    pop_size,
    0.9,
    0.8,
    0.5,
    x_min=x_min,
    x_max=x_max,
    gd_alpha=1e-4,
)


s = np.linspace(x_min, x_max, 100)
tests = tf.reshape(tf.Variable(s, dtype="float32"), [100, 1])
results = objective(tests)


def snapshot(i):

    opt_pso.train()
    nn_w_pso, nn_b_pso = opt_pso.get_best()
    pred_pso = utils.multilayer_perceptron(nn_w_pso, nn_b_pso, tests)

    opt_pso_gd.train()
    nn_w_pso_gd, nn_b_pso_gd = opt_pso_gd.get_best()
    pred_pso_gd = utils.multilayer_perceptron(nn_w_pso_gd, nn_b_pso_gd, tests)

    opt_pso_gd2.train()
    nn_w_pso_gd2, nn_b_pso_gd2 = opt_pso_gd2.get_best()
    pred_pso_gd2 = utils.multilayer_perceptron(nn_w_pso_gd2, nn_b_pso_gd2, tests)

    plt.clf()
    plt.xlim([-1, 1])
    plt.ylim([-1.5, 1.5])

    plt.plot(
        tf.squeeze(tests), tf.squeeze(results), label="$f(x) = x^3 - \sin (2 \pi x)$"
    )
    plt.plot(tf.squeeze(tests), tf.squeeze(pred_pso), "--", label="PSO")
    plt.plot(
        tf.squeeze(tests),
        tf.squeeze(pred_pso_gd),
        "--",
        label="PSO - GD (lr = 1e-2)",
    )
    plt.plot(
        tf.squeeze(tests),
        tf.squeeze(pred_pso_gd2),
        "--",
        label="PSO - GD (lr= 1e-4)",
    )
    plt.text(0.7, -1.2, "it: " + str(i * n_iter), fontsize=14)
    plt.text(-0.2, -1.4, "ANN layers: " + str(layer_sizes), fontsize=14)
    plt.legend()


fig = plt.figure(figsize=(8, 8), dpi=100)
anim = animation.FuncAnimation(fig, snapshot, frames=60)
anim.save("swarm_training_PSOGD.gif", fps=6)
