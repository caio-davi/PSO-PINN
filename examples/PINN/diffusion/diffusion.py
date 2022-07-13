import numpy as np
import tensorflow as tf
import argparse
import time

from pyDOE import lhs

import swarm.optimizers as optimizers


import matplotlib.pyplot as plt
from swarm.utils import multilayer_perceptron, decode
import matplotlib.animation as animation


from scipy.interpolate import griddata


# define grid for solution
utn = 256
uxn = 1024
xlo = 0
xhi = 1
ux = np.linspace(xlo, xhi, uxn)
tlo = 0
thi = 1
ut = np.linspace(tlo, thi, utn)


def difusion_eq_exact_solution(x, t):
    return (
        1
        + t
        + np.exp(-4 * (np.pi**2) * t) * np.cos(2 * np.pi * x)
        + x * np.sin(t)
    )


u_quad = []
for utj in ut:
    ux_t = np.linspace(xlo, xhi, uxn)
    u_quad.append(difusion_eq_exact_solution(ux_t, utj))

u_quad = np.array(u_quad).T

# collocation points
Ncl = 512
X = lhs(2, Ncl)
xcl = tf.expand_dims(
    tf.convert_to_tensor(xlo + (xhi - xlo) * X[:, 0], dtype=tf.float32), -1
)
tcl = tf.expand_dims(
    tf.convert_to_tensor(tlo + (thi - tlo) * X[:, 1], dtype=tf.float32), -1
)

# initial condition points
x0 = tf.expand_dims(tf.convert_to_tensor(ux, dtype=tf.float32), -1)
t0 = tf.zeros(tf.shape(x0), dtype=tf.float32)
q0 = tf.convert_to_tensor(1 + np.cos(2 * np.pi * x0), dtype=tf.float32)

# Dirichlet boundary condition points
xlb = tf.expand_dims(xlo * tf.ones(tf.shape(ut), dtype=tf.float32), -1)
tlb = tf.expand_dims(tf.convert_to_tensor(ut, dtype=tf.float32), -1)

xub = tf.expand_dims(xhi * tf.ones(tf.shape(ut), dtype=tf.float32), -1)
tub = tf.expand_dims(tf.convert_to_tensor(ut, dtype=tf.float32), -1)

qb = tf.convert_to_tensor(np.sin(tlb), dtype=tf.float32)

# residual
def get_residual(x, t):
    return 1 + x * np.cos(t)


res = []
for tcli in tcl:
    res.append(get_residual(xcl, tcli))

res = np.asarray(res)


@tf.function
def f(w, b, x, t):
    u = multilayer_perceptron(w, b, tf.concat([x, t], 1))
    u_x = tf.gradients(u, x)[0]
    u_xx = tf.gradients(u_x, x)[0]
    u_t = tf.gradients(u, t)[0]
    f_u = u_t - u_xx - res
    return f_u


@tf.function
def fx(w, b, x, t):
    u = multilayer_perceptron(w, b, tf.concat([x, t], 1))
    u_x = tf.gradients(u, x)[0]
    return u_x


def loss(w, b):
    pred_0 = multilayer_perceptron(w, b, tf.concat([x0, t0], 1))
    pred_x0 = fx(w, b, xlb, tlb)
    pred_x1 = fx(w, b, xub, tub)
    fr = f(w, b, xcl, tcl)

    # IC loss
    mse_0 = tf.reduce_mean(tf.pow(q0 - pred_0, 2))

    # Dirichlet boundary loss
    mse_lb = tf.reduce_mean(tf.pow(qb - pred_x0, 2))
    mse_ub = tf.reduce_mean(tf.pow(qb - pred_x1, 2))

    # Residual loss
    mse_f = tf.reduce_mean(tf.pow(fr, 2))

    return mse_0 + mse_f + mse_lb + mse_ub


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


optmizer = "pso_adam"
hidden_layers = 5
neurons_per_layer = 8
layer_sizes = [2] + hidden_layers * [neurons_per_layer] + [1]
n_iter = 4000


def run_swarm(swarm, X):
    new_swarm = []
    for particle in swarm:
        w, b = decode(particle, layer_sizes)
        new_swarm.append(multilayer_perceptron(w, b, X))
    return new_swarm


X, T = np.meshgrid(ux, ut)
X_flat = tf.convert_to_tensor(
    np.hstack((X.flatten()[:, None], T.flatten()[:, None])),
    dtype=tf.float32,
)
q_flat = u_quad.T.flatten()

opt = optimizers.get[optmizer](loss_grad(), layer_sizes, n_iter)

start = time.time()
opt.train()
end = time.time()
swarm = opt.get_swarm()
preds = run_swarm(swarm, X_flat)
mean = tf.squeeze(tf.reduce_mean(preds, axis=0))
var = tf.squeeze(tf.math.reduce_std(preds, axis=0))
err = np.linalg.norm(q_flat - mean, 2) / np.linalg.norm(q_flat, 2)

time_steps = utn  # total number of time steps in animation
fps = 15  # frames/second of animation


def snapshot(i):
    l_ind = i * uxn
    u_ind = (i + 1) * uxn
    plt.clf()
    plt.ylim([0, 3])
    plt.plot(ux, q_flat[l_ind:u_ind], "b-", linewidth=3, label="Quadrature")
    locs, labels = plt.xticks()
    plt.plot(
        ux,
        mean[l_ind:u_ind],
        "r--",
        linewidth=3,
        label=opt.name,
    )
    plt.fill_between(
        ux,
        mean[l_ind:u_ind] - 3 * var[l_ind:u_ind],
        mean[l_ind:u_ind] + 3 * var[l_ind:u_ind],
        color="gray",
        alpha=0.2,
    )
    plt.xlabel("$x$", fontsize="xx-large")
    plt.ylabel("$u(t,x)$", fontsize="xx-large")
    plt.grid()
    plt.legend(fontsize="x-large")


fig = plt.figure(figsize=(8, 8), dpi=150)
anim = animation.FuncAnimation(fig, snapshot, frames=time_steps)
anim.save("Diffusion_Demo.gif", fps=fps)
