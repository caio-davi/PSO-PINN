import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import numpy as np
import math
from swarm.optimizers.pso import pso
from swarm.utils import multilayer_perceptron, decode, replacenan

np.random.seed(123456)
tf.random.set_seed(123456)

uxn = 256 + 2
xlo = 0
xhi = 1

ux = tf.reshape(tf.Variable(np.linspace(xlo, xhi, uxn), dtype="float32"), [uxn, 1])

u = tf.sin(2 * math.pi * ux)

x0 = tf.reshape(tf.convert_to_tensor(xlo, dtype=tf.float32), [1, 1])
x1 = tf.reshape(tf.convert_to_tensor(xhi, dtype=tf.float32), [1, 1])
u0 = tf.reshape(tf.convert_to_tensor(u[0], dtype=tf.float32), [1, 1])
u1 = tf.reshape(tf.convert_to_tensor(u[-1], dtype=tf.float32), [1, 1])

layer_sizes = [1] + 3 * [10] + [1]
pop_size = 100
n_iter = 30
stepInd = 0.01
stepVol = 0.01
w_scale = 100

def objective(x):
    return tf.sin(2 * math.pi * x)

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

test_size = 100
s = np.linspace(xlo, xhi, test_size)
X = tf.reshape(tf.Variable(s, dtype="float32"), [test_size, 1])

opt = pso(
    loss_grad(),
    layer_sizes,
    n_iter,
    pop_size,
    0.9,
    0.08,
    0.05,
    verbose=True,
    gd_alpha=1e-5,
)

def nn_wrapper(particle):
    w, b = decode(particle, layer_sizes)
    return multilayer_perceptron(w, b, X)

steps = 3
mpl.style.use("seaborn")
fig = plt.figure(figsize=(15, 7), dpi=300)
plt.subplots_adjust(hspace=0.5)
ax = []

for i in range(2):
    temp = []
    for j in range(steps):
        temp.append(plt.subplot2grid((3, 3), (i, j)))
    ax.append(temp)
ax.append(plt.subplot2grid((3, 3), (2, 0), colspan=3))


it_counter = n_iter
for j in range(steps):
    start = time.time()
    opt.train()
    end = time.time()

    swarm = opt.get_swarm()
    preds = tf.reshape(tf.vectorized_map(nn_wrapper, swarm), [pop_size, test_size])

    mean = tf.reduce_mean(replacenan(preds), axis=0)
    variance = tf.math.reduce_variance(replacenan(preds), axis=0)

    nn_w, nn_b = opt.get_best()
    pred = multilayer_perceptron(nn_w, nn_b, X)
    X_ = tf.squeeze(X)
    y = objective(X)

    for k in range(preds.shape[0]):
        ax[0][j].plot(X_, preds[k], linewidth=0.2)
    ax[1][j].fill_between(
        X_,
        mean - variance,
        mean + variance,
        color="gray",
        alpha=0.5,
    )
    ax[1][j].plot(X_, y, label="Original Function")
    ax[1][j].plot(X_, pred, "--", label="Best")
    ax[1][j].plot(X_, mean, "--", label="Mean")
    ax[0][j].set_title(str(it_counter) + " iterations", fontsize="xx-large")
    ax[1][j].legend()

    it_counter = it_counter + n_iter


ax[2].plot(opt.loss_history)
ax[2].set_title("Loss Error (mean)", fontsize="xx-large")

plt.savefig("poisson_train.png", bbox_inches="tight")
