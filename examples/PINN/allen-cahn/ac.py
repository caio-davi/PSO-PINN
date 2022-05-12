import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle

from swarm.optimizers.pso_adam import pso
from swarm.utils import multilayer_perceptron, decode
import matplotlib.pyplot as plt
import matplotlib.animation as animation


np.random.seed(1)
tf.random.set_seed(1)


f = open("AC.pckl", "rb")
data = pickle.load(f)
f.close()

t = np.array(data["t"]).flatten()[:, None]
x = np.array(data["x"]).flatten()[:, None]
Exact_u = np.array(data["u"])

## Positive half of the problem
# x = x[256:]
# Exact_u = Exact_u[256:]

uxn = len(x)
utn = len(t)


lb = np.array([-1.0])
ub = np.array([1.0])

N0 = 256
N_b = 200
N_f = 500

data_size = 10


idx_x = np.random.choice(x.shape[0], N0, replace=False)
x0 = x[idx_x, :]
u0 = Exact_u[idx_x, 0:1]

# u0 = tf.cast(u0, tf.float3)
idx_t = np.random.choice(t.shape[0], N_b, replace=False)
tb = t[idx_t, :]

x_f_idx = np.random.choice(x.shape[0], N_f, replace=True)
t_f_idx = np.random.choice(t.shape[0], N_f, replace=True)

x_f = tf.convert_to_tensor(x[x_f_idx, :], dtype=tf.float32)
t_f = tf.convert_to_tensor(t[t_f_idx, :], dtype=tf.float32)
u_f = tf.convert_to_tensor(Exact_u[x_f_idx, t_f_idx], dtype=tf.float32)

X0 = np.concatenate((x0, 0 * x0), 1)  # (x0, 0)
X_lb = np.concatenate((0 * tb + lb[0], tb), 1)  # (lb[0], tb)
X_ub = np.concatenate((0 * tb + ub[0], tb), 1)  # (ub[0], tb)

x0 = X0[:, 0:1]
t0 = X0[:, 1:2]
X_0 = tf.cast(tf.concat([x0, t0], 1), dtype=tf.float32)

x_lb = tf.convert_to_tensor(X_lb[:, 0:1], dtype=tf.float32)
t_lb = tf.convert_to_tensor(X_lb[:, 1:2], dtype=tf.float32)

x_ub = tf.convert_to_tensor(X_ub[:, 0:1], dtype=tf.float32)
t_ub = tf.convert_to_tensor(X_ub[:, 1:2], dtype=tf.float32)


splits = np.linspace(
    int(0.15 * uxn), int(0.85 * uxn), data_size, endpoint=False
).astype(int)

x_d = x[splits, :][:, None]
X_D, T_D = np.meshgrid(x_d, t)
X_D = np.hstack((X_D.flatten()[:, None], T_D.flatten()[:, None]))

u_d = Exact_u[splits]
u_d_flat = u_d.T.flatten()[:, None]


def f_model(w, b, x, t):
    # keep track of our gradients
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        tape.watch(t)

        # Getting the prediction
        u = multilayer_perceptron(w, b, tf.concat([x, t], 1))
        # _, u_x = u_x_model(x, t)
        u_x = tape.gradient(u, x)
    # Getting the other derivatives
    u_xx = tape.gradient(u_x, x)
    u_t = tape.gradient(u, t)

    # Letting the tape go
    del tape

    f_u = u_t - 0.0001 * u_xx + 5.0 * u**3 - 5.0 * u

    return f_u


def u_x_model(w, b, x, t):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        tape.watch(t)
        X = tf.concat([x, t], 1)
        u = multilayer_perceptron(w, b, X)
        # u = u[:, 0:1]

    u_x = tape.gradient(u, x)

    del tape

    return u, u_x


@tf.function
def loss(w, b):

    f_u_pred = f_model(w, b, x_f, t_f)

    u0_pred = multilayer_perceptron(w, b, X_0)
    u_lb_pred, u_x_lb_pred = u_x_model(w, b, x_lb, t_lb)
    u_ub_pred, u_x_ub_pred = u_x_model(w, b, x_ub, t_ub)

    u_d_pred = multilayer_perceptron(w, b, X_D.astype(np.float32))

    mse_u = tf.reduce_mean(tf.square(u0 - u0_pred))

    mse_b = tf.reduce_mean(tf.square(u_lb_pred - u_ub_pred)) + tf.reduce_mean(
        tf.square(u_x_lb_pred + u_x_ub_pred)
    )

    mse_f = tf.reduce_mean(tf.square(f_u_pred))

    mse_d = tf.reduce_mean(tf.square(u_d_flat - u_d_pred))

    # return 100 * mse_u + mse_b + mse_f + 20 * mse_d
    return 10 * mse_u + 10 * mse_d + mse_f + mse_b


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


def run_swarm(swarm, X):
    swarm_y = []
    for particle in swarm:
        w, b = decode(particle, layer_sizes)
        swarm_y.append(multilayer_perceptron(w, b, X))
    return swarm_y


def format_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return h, m, s


layer_sizes = [2] + 5 * [15] + [1]
pop_size = 10
n_iter = 6000

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
print("Time elapsed : %2d:%2d:%2d" % format_time(end - start))


X, T = np.meshgrid(x, t)
X_flat = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
u_star = Exact_u.T.flatten()
x_ = np.squeeze(x)
swarm = opt.get_swarm()
preds = run_swarm(swarm, X_flat.astype(np.float32))
mean = tf.squeeze(tf.reduce_mean(preds, axis=0))
var = tf.squeeze(tf.math.reduce_std(preds, axis=0))
print("Last Loss: ", opt.loss_history[-1])

time_steps = utn - 1  # total number of time steps in animation
fps = 15  # frames/second of animation
error_u = np.linalg.norm(u_star - mean, 2) / np.linalg.norm(u_star, 2)
print("Error u: %e" % (error_u))


def snapshot(i):
    l_ind = i * uxn
    u_ind = (i + 1) * uxn
    plt.clf()
    for k in range(len(preds)):
        plt.plot(x_, preds[k][l_ind:u_ind], linewidth=0.3)
    plt.scatter(x_d, u_d_flat[i * data_size : (i + 1) * data_size])
    plt.plot(x_, u_star[l_ind:u_ind], "b-", linewidth=3, label="Exact")
    plt.plot(
        x_,
        mean[l_ind:u_ind],
        "r--",
        linewidth=3,
        label=opt.name,
    )
    plt.fill_between(
        x_,
        mean[l_ind:u_ind] - 3 * var[l_ind:u_ind],
        mean[l_ind:u_ind] + 3 * var[l_ind:u_ind],
        color="gray",
        alpha=0.2,
    )

    plt.xlabel("$x$", fontsize="xx-large")
    plt.ylabel("$u(t,x)$", fontsize="xx-large")
    plt.xlim(-1.0, 1.0)
    plt.ylim(-1.02, 1.02)
    plt.grid()
    plt.legend(fontsize="x-large")


fig = plt.figure(figsize=(8, 8), dpi=150)
# Call the animator:
anim = animation.FuncAnimation(fig, snapshot, frames=time_steps)
# Save the animation as an mp4. This requires ffmpeg to be installed.
anim.save("ac_demo.gif", fps=fps)
