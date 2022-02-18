import numpy as np
import tensorflow as tf
from pyDOE import lhs
import matplotlib as mpl
import matplotlib.pyplot as plt

from swarm.optimizers.pso import pso
from swarm.utils import multilayer_perceptron

np.random.seed(123456)
tf.random.set_seed(123456)

# parameters of the simulation
ql = 4  # concentration at the left
qm1 = 0.4  # intermediate concentration
qm2 = 4  # intermediate concentration
qm3 = 1.4  # intermediate concentration
qm4 = 3  # intermediate concentration
qm5 = 0.4  # intermediate concentration
qm6 = 4  # intermediate concentration
qr = 0.4  # concentration at the right
x0 = 1  # central discontinuity location
x01 = 0.25  # left discontinuity location
x02 = 0.5  # right discontinuity location
x03 = 0.75  # left discontinuity location
x04 = 1.25  # right discontinuity location
x05 = 1.5  # left discontinuity location
x06 = 1.75  # right discontinuity location
L = 2  # channel length
u = 1  # advection speed

# nu = 0.01 # viscosity parameter

# define grid for solution
utn = 256
uxn = 1024
xlo = 0
xhi = L
ux = np.linspace(xlo, xhi, uxn)
tlo = 0.0
thi = 0.2
ut = np.linspace(tlo, thi, utn)

# analytical solution basic step
q = np.zeros([uxn, utn])
for utj in range(utn):
    j0 = int(uxn * (x0 + u * ut[utj]) / L)
    for uxj in range(0, j0 + 1):
        q[uxj, utj] = ql
    for uxj in range(j0 + 1, uxn):
        q[uxj, utj] = qr

layer_sizes = [2] + 5 * [10] + [1]
pop_size = 100
n_iter = 100


# collocation points
Ncl = 5000
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
q0 = tf.expand_dims(tf.convert_to_tensor(q[:, 0], dtype=tf.float32), -1)

# Dirichlet boundary condition points
xlb = tf.expand_dims(xlo * tf.ones(tf.shape(ut), dtype=tf.float32), -1)
tlb = tf.expand_dims(tf.convert_to_tensor(ut, dtype=tf.float32), -1)
qlb = ql * tf.ones(tf.shape(tlb), dtype=tf.float32)

xub = tf.expand_dims(xhi * tf.ones(tf.shape(ut), dtype=tf.float32), -1)
tub = tf.expand_dims(tf.convert_to_tensor(ut, dtype=tf.float32), -1)
qub = qr * tf.ones(tf.shape(tub), dtype=tf.float32)


def f(w, b, x, t):
    q = multilayer_perceptron(w, b, tf.concat([x, t], 1))  # compute q

    q_t = tf.gradients(q, t)[0]
    q_x = tf.gradients(q, x)[0]

    fr = q_t + u * q_x

    return fr


@tf.function
def loss(w, b):

    pred_0 = multilayer_perceptron(w, b, tf.concat([x0, t0], 1))
    pred_lb = multilayer_perceptron(w, b, tf.concat([xlb, tlb], 1))
    pred_ub = multilayer_perceptron(w, b, tf.concat([xub, tub], 1))
    fr = f(w, b, xcl, tcl)

    # IC loss
    mse_0 = tf.reduce_mean(tf.pow(q0 - pred_0, 2))

    # Dirichlet boundary loss
    mse_lb = tf.reduce_mean(tf.pow(qlb - pred_lb, 2))
    mse_ub = tf.reduce_mean(tf.pow(qub - pred_ub, 2))

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


opt = pso(
    loss_grad(),
    layer_sizes,
    n_iter,
    pop_size,
    0.9,
    0.08,
    0.05,
    verbose=True,
    gd_alpha=1e-2,
)

opt.train()

X, T = np.meshgrid(ux, ut)
X_flat = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
nn_w, nn_b = opt.get_best()
u_pso = multilayer_perceptron(nn_w, nn_b, X_flat.astype(np.float32))

q_flat = q.T.flatten()
err_pso = np.linalg.norm(q_flat - tf.squeeze(u_pso), 2) / np.linalg.norm(q_flat, 2)
print("L2 error PSO-GD: %.2e" % (err_pso))

mpl.style.use("seaborn")
fig = plt.figure(figsize=(8, 8), dpi=300)
plt.subplots_adjust(hspace=0.3)
ax = fig.subplots(3)
m = [0, int(utn / 2), utn - 1]

for i in [0, 1, 2]:
    ax[i].plot(q[:, m[i]], "-", linewidth=3, label="Analytical")
    ax[i].plot(
        u_pso[m[i] * uxn : (m[i] + 1) * uxn], "--", linewidth=3, label="PSO-PINN"
    )
    ax[i].set_title(r"t = {:.2f}/$\pi$".format(np.pi * m[i] * thi / (utn - 1)))
    ax[i].set_ylabel("$u(t,x)$")
    ax[i].legend()
ax[2].set_xlabel("$x$", fontsize="xx-large")

plt.savefig("advection.png", bbox_inches="tight")
