import tensorflow as tf
import matplotlib.pyplot as plt
import time
import numpy as np
import math

from swarm.optimizers.pso import pso
from swarm import utils

np.random.seed(123456)
tf.random.set_seed(123456)


# Parameters
layers = [1] + 3 * [5] + [1]
pop_size = 100
n_iter = 2000
x_min = -1
x_max = 1
sample_size = 512
noise = 0.0


def objective(x, noise=0):
    return tf.cos(math.pi * x) - x

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

y_min, y_max = tf.math.reduce_min(y), tf.math.reduce_max(y)

X, y = utils.normalize(X, [x_min, x_max]), utils.normalize(y, [y_min, y_max])

opt = pso(
    get_loss(X, y),
    layers,
    n_iter,
    pop_size,
    0.9,
    0.8,
    0.5,
    gd_alpha=1e-4,
    x_min=x_min,
    x_max=x_max,
    verbose=True,
)

start = time.time()
opt.train()
end = time.time()
print("\nTime elapsed: ", end - start)

nn_w, nn_b = opt.get_best()

pred = utils.multilayer_perceptron(nn_w, nn_b, X)

print("L2 error: ", tf.reduce_mean(tf.pow(y - pred, 2)).numpy())

plt.plot(tf.squeeze(X), tf.squeeze(y), label="Original Function")
plt.plot(tf.squeeze(X), tf.squeeze(pred), "--", label="Swarm")
plt.legend()
plt.show()
