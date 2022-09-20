import numpy as np
from numpy.polynomial.hermite import hermgauss
import tensorflow as tf
import time
from pyDOE import lhs
from swarm.optimizers.pso_adam import pso
from swarm.utils import multilayer_perceptron, decode
import matplotlib.pyplot as plt
import matplotlib.animation as animation

np.random.seed(12345)
tf.random.set_seed(12345)


"""
Problem Definition and Quadrature Solution
"""

nu = 0.01  # / np.pi  # viscosity parameter

# define grid for quadrature solution
utn = 100
uxn = 256
xlo = -1.0
xhi = +1.0
ux = np.linspace(xlo, xhi, uxn)
tlo = 0.0
thi = 1.0
ut = np.linspace(tlo, thi, utn)

qn = 32  # order of quadrature rule
qx, qw = hermgauss(qn)

# compute solution u(x,t) by quadrature of analytical formula:
u_quad = np.zeros([uxn, utn])
for utj in range(utn):
    if ut[utj] == 0.0:
        for uxj in range(uxn):
            u_quad[uxj, utj] = -np.sin(np.pi * ux[uxj])
    else:
        for uxj in range(uxn):
            top = 0.0
            bot = 0.0
            for qj in range(qn):
                c = 2.0 * np.sqrt(nu * ut[utj])
                top = top - qw[qj] * c * np.sin(
                    np.pi * (ux[uxj] - c * qx[qj])
                ) * np.exp(
                    -np.cos(np.pi * (ux[uxj] - c * qx[qj]))
                    / (2.0 * np.pi * nu)
                )
                bot = bot + qw[qj] * c * np.exp(
                    -np.cos(np.pi * (ux[uxj] - c * qx[qj]))
                    / (2.0 * np.pi * nu)
                )
                u_quad[uxj, utj] = top / bot


# Algorithm parameters
layer_sizes = [2] + 5 * [15] + [1]
pop_size = 10
n_iter = 6000


# collocation points
def collocation_points(size):
    X = lhs(2, size)
    xcl = tf.expand_dims(
        tf.convert_to_tensor(xlo + (xhi - xlo) * X[:, 0], dtype=tf.float32), -1
    )
    tcl = tf.expand_dims(
        tf.convert_to_tensor(tlo + (thi - tlo) * X[:, 1], dtype=tf.float32), -1
    )
    return xcl, tcl


# initial condition points
x0 = tf.expand_dims(tf.convert_to_tensor(ux, dtype=tf.float32), -1)
t0 = tf.zeros(tf.shape(x0), dtype=tf.float32)
u0 = -tf.math.sin(np.pi * x0)

# Dirichlet boundary condition points
xlb = tf.expand_dims(xlo * tf.ones(tf.shape(ut), dtype=tf.float32), -1)
tlb = tf.expand_dims(tf.convert_to_tensor(ut, dtype=tf.float32), -1)
ulb = tf.expand_dims(tf.zeros(tf.shape(ut), dtype=tf.float32), -1)
xub = tf.expand_dims(xhi * tf.ones(tf.shape(ut), dtype=tf.float32), -1)
tub = tf.expand_dims(tf.convert_to_tensor(ut, dtype=tf.float32), -1)
uub = tf.expand_dims(tf.zeros(tf.shape(ut), dtype=tf.float32), -1)


@tf.function
def f_model(w, b):
    x, t = collocation_points(500)
    u = multilayer_perceptron(w, b, tf.concat([x, t], 1))
    u_x = tf.gradients(u, x)
    u_xx = tf.gradients(u_x, x)
    u_t = tf.gradients(u, t)
    f_u = u_t + u * u_x - [nu * element for element in u_xx]
    return f_u


@tf.function
def loss(w, b):
    u0_pred = multilayer_perceptron(w, b, tf.concat([x0, t0], 1))
    ulb_pred = multilayer_perceptron(w, b, tf.concat([xlb, tlb], 1))
    uub_pred = multilayer_perceptron(w, b, tf.concat([xub, tub], 1))
    f_pred = f_model(w, b)

    # IC loss
    mse_0 = tf.reduce_mean(tf.pow(u0 - u0_pred, 2))

    # Dirichlet BC loss
    mse_lb = tf.reduce_mean(tf.pow(ulb_pred - ulb, 2))
    mse_ub = tf.reduce_mean(tf.pow(uub_pred - uub, 2))

    # Residual loss
    mse_f = tf.reduce_mean(tf.pow(f_pred, 2))

    return 1.5 * mse_0 + mse_f + mse_lb + mse_ub


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


def run_swarm(swarm, X_flat):
    new_swarm = []
    for particle in swarm:
        w, b = decode(particle, layer_sizes)
        new_swarm.append(
            multilayer_perceptron(w, b, X_flat.astype(np.float32))
        )
    return tf.convert_to_tensor(new_swarm, dtype=tf.float32)


opt = pso(
    loss_grad(),
    layer_sizes,
    n_iter,
    pop_size,
    0.999,
    8e-4,
    5e-3,
    initialization_method="xavier",
    verbose=True,
    gd_alpha=1e-4,
)

start = time.time()
opt.train()
end = time.time()
print("\nTime elapsed: ", end - start)

X, T = np.meshgrid(ux, ut)
X_flat = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

nn_w, nn_b = opt.get_best()
best = multilayer_perceptron(nn_w, nn_b, X_flat.astype(np.float32))

swarm = opt.get_swarm()
preds = run_swarm(swarm, X_flat.astype(np.float32))
mean = tf.squeeze(tf.reduce_mean(preds, axis=0))
var = tf.squeeze(tf.math.reduce_std(preds, axis=0))

u_PINN = multilayer_perceptron(nn_w, nn_b, X_flat.astype(np.float32))

"""
Comparison and Animation
"""

u_quad_flat = u_quad.T.flatten()[:, None]
error_u = np.linalg.norm(u_quad_flat - u_PINN, 2) / np.linalg.norm(
    u_quad_flat, 2
)
print("nu = %.2f/pi" % (np.pi * nu))
print("L2 Error = %e" % (error_u))
print("Last Loss: ", opt.loss_history[-1])

time_steps = utn  # total number of time steps in animation
fps = 15  # frames/second of animation


def snapshot(i):
    plt.clf()
    plt.plot(ux, u_quad[:, i], "b-", linewidth=3, label="Quadrature")
    plt.plot(
        ux,
        mean[i * uxn : (i + 1) * uxn],
        "r--",
        linewidth=3,
        label="PSO-PINN",
    )
    plt.fill_between(
        ux,
        mean[i * uxn : (i + 1) * uxn] - var[i * uxn : (i + 1) * uxn],
        mean[i * uxn : (i + 1) * uxn] + var[i * uxn : (i + 1) * uxn],
        color="gray",
        alpha=0.5,
    )
    plt.xlabel("$x$", fontsize="xx-large")
    plt.ylabel("$u(t,x)$", fontsize="xx-large")
    plt.grid()
    plt.xlim(-1.02, 1.02)
    plt.ylim(-1.02, 1.02)
    plt.legend(fontsize="x-large")


fig = plt.figure(figsize=(8, 8), dpi=150)
# Call the animator:
anim = animation.FuncAnimation(fig, snapshot, frames=time_steps)
# Save the animation as an mp4. This requires ffmpeg to be installed.
anim.save("Burgers_Demo.gif", fps=fps)
