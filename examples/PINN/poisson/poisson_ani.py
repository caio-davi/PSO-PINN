import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math

from swarm.optimizers.pso import pso

from swarm.utils import multilayer_perceptron


np.random.seed(1234)
tf.random.set_seed(1234)

uxn = 256 + 2
xlo = 0
xhi = 1

ux = tf.reshape(tf.Variable(np.linspace(xlo, xhi, uxn), dtype="float32"), [uxn, 1])

u = tf.sin(2 * math.pi * ux)

x0 = tf.reshape(tf.convert_to_tensor(xlo, dtype=tf.float32), [1, 1])
x1 = tf.reshape(tf.convert_to_tensor(xhi, dtype=tf.float32), [1, 1])
u0 = tf.reshape(tf.convert_to_tensor(u[0], dtype=tf.float32), [1, 1])
u1 = tf.reshape(tf.convert_to_tensor(u[-1], dtype=tf.float32), [1, 1])

layer_sizes = [1] + 3 * [20] + [1]
pop_size = 100
n_iter = 10
stepInd = 0.01
stepVol = 0.01
w_scale = 100
x_min = -1
x_max = 1
sample_size = 10


noise = 0.0


def objective(x, noise=0):
    return (tf.sin(2 * math.pi * x)) + tf.random.normal(x.shape, 0, noise, tf.float32)


def u2(x):
    return tf.cast(-4 * np.pi * np.pi * tf.sin(2 * np.pi * x), dtype=tf.float32)


@tf.function
def r(w, b):
    q = multilayer_perceptron(w, b, ux)
    q_x = tf.gradients(q, ux)[0]
    q_xx = tf.gradients(q_x, ux)[0]
    return tf.subtract(q_xx, u2(ux))


@tf.function
def loss(w, b):
    pred_0 = multilayer_perceptron(w, b, x0)
    pred_1 = multilayer_perceptron(w, b, x1)
    pred_r = r(w, b)

    mse_0 = 100 * tf.pow(u0 - pred_0, 2)
    mse_1 = 100 * tf.pow(u1 - pred_1, 2)
    mse_r = tf.pow(pred_r, 2)

    return tf.reduce_mean(mse_0 + mse_1 + mse_r)


def loss_grad():
    def _loss(w, b):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(w)
            tape.watch(b)
            loss_value = loss(w, b)

        trainable_variables = w + b
        grads = tape.gradient(loss_value, trainable_variables)
        return loss_value, grads

    return _loss


opt_pso = pso(
    loss_grad(),
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

opt_pso_gd_1 = pso(
    loss_grad(),
    layer_sizes,
    n_iter,
    pop_size,
    0.9,
    0.8,
    0.5,
    x_min=x_min,
    x_max=x_max,
    gd_alpha=1e-3,
)

opt_pso_gd_2 = pso(
    loss_grad(),
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

opt_pso_gd_3 = pso(
    loss_grad(),
    layer_sizes,
    n_iter,
    pop_size,
    0.9,
    0.8,
    0.5,
    x_min=x_min,
    x_max=x_max,
    gd_alpha=1e-5,
)


s = np.linspace(xlo, xhi, 100)
tests = tf.reshape(tf.Variable(s, dtype="float32"), [100, 1])
results = objective(tests)


def snapshot(i):

    opt_pso.train()
    nn_w_pso, nn_b_pso = opt_pso.get_best()
    pred_pso = multilayer_perceptron(nn_w_pso, nn_b_pso, tests)

    opt_pso_gd_1.train()
    nn_w_pso_1, nn_b_pso_1 = opt_pso_gd_1.get_best()
    pred_pso_1 = multilayer_perceptron(nn_w_pso_1, nn_b_pso_1, tests)

    opt_pso_gd_2.train()
    nn_w_pso_2, nn_b_pso_2 = opt_pso_gd_2.get_best()
    pred_pso_2 = multilayer_perceptron(nn_w_pso_2, nn_b_pso_2, tests)

    opt_pso_gd_3.train()
    nn_w_pso_3, nn_b_pso_3 = opt_pso_gd_3.get_best()
    pred_pso_3 = multilayer_perceptron(nn_w_pso_3, nn_b_pso_3, tests)

    plt.clf()
    plt.xlim([0, 1])
    plt.ylim([-1.5, 1.5])

    plt.plot(tf.squeeze(tests), tf.squeeze(results), label="$f(x) = \sin (2 \pi x)$")
    plt.plot(tf.squeeze(tests), tf.squeeze(pred_pso), "--", label="PSO ")
    plt.plot(tf.squeeze(tests), tf.squeeze(pred_pso_1), "--", label="PSO GD (lr: 1e-3)")
    plt.plot(
        tf.squeeze(tests), tf.squeeze(pred_pso_2), "--", label="PSO - GD (lr: 1e-4)"
    )
    plt.plot(
        tf.squeeze(tests),
        tf.squeeze(pred_pso_3),
        "--",
        label="PSO - GD (lr: 1e-5)",
    )
    plt.text(0.7, -1.2, "it: " + str(i * n_iter), fontsize=14)
    plt.legend()


fig = plt.figure(figsize=(8, 8), dpi=100)
anim = animation.FuncAnimation(fig, snapshot, frames=40)
anim.save("poisson_PSO.gif", fps=2)
