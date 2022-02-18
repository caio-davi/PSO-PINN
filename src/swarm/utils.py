import tensorflow as tf


def dimensions(layers):
    sum = 0
    for l in range(0, len(layers) - 1):
        sum += layers[l] * layers[l + 1] + layers[l + 1]
    return sum


def progress(percent=0, width=30, metric=None, metricValue=None):
    left = width * int(percent) // 100
    right = width - left
    if metric:
        print(
            "\r[",
            "#" * left,
            " " * right,
            "]",
            f" {percent:.0f}%  --  ",
            "Current ",
            metric,
            ": ",
            metricValue,
            sep="",
            end="",
            flush=True,
        )
    else:
        print(
            "\r[",
            "#" * left,
            " " * right,
            "]",
            f" {percent:.0f}%  --  ",
            sep="",
            end="",
            flush=True,
        )


def decode(encoded, layers):
    weights = []
    biases = []
    last_cut = 0
    num_layers = len(layers)
    for l in range(0, num_layers - 1):
        next_cut = layers[l] * layers[l + 1]
        W = tf.reshape(
            tf.slice(encoded, [last_cut], [next_cut]), [layers[l], layers[l + 1]]
        )
        last_cut = last_cut + next_cut
        weights.append(W)
    for l in range(1, num_layers):
        b = tf.slice(encoded, [last_cut], [layers[l]])
        last_cut += layers[l]
        biases.append(b)
    return weights, biases


def multilayer_perceptron(weights, biases, X, x_min=-1, x_max=1):
    num_layers = len(weights) + 1
    H = 2.0 * (X - x_min) / (x_max - x_min) - 1.0
    for l in range(0, num_layers - 2):
        W = weights[l]
        b = biases[l]
        H = tf.nn.tanh(tf.add(tf.matmul(H, W), b))
    W = weights[-1]
    b = biases[-1]
    Y = tf.add(tf.matmul(H, W), b)
    return Y


def replacenan(t):
    return tf.where(tf.math.is_nan(t), tf.zeros_like(t), t)

def clip(t):
    return tf.clip_by_value(t,1e-10,100)


def xavier_init(size):
    in_dim = size[0]
    out_dim = size[1]
    xavier_stddev = tf.sqrt(2 / (in_dim + out_dim))
    return tf.Variable(
        tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev),
        dtype=tf.float32,
    )


def initialize_NN(layers):
    weights = []
    biases = []
    num_layers = len(layers)
    for l in range(0, num_layers - 1):
        W = xavier_init(size=[layers[l], layers[l + 1]])
        b = tf.Variable(
            tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32
        )
        weights.append(W)
        biases.append(b)
    return weights, biases


def make_pop_NN(pop_size, layer_sizes):
    xavier_init_nns = []
    for _ in range(pop_size):
        w, b = initialize_NN(layer_sizes)
        new_nn = encode(w, b)
        xavier_init_nns.append(new_nn)
    return tf.Variable(xavier_init_nns, dtype=tf.float32)


def encode(weights, biases):
    encoded = tf.Variable([])
    for l in weights:
        encoded = tf.concat(values=[encoded, tf.reshape(l, -1)], axis=-1)
    for l in biases:
        encoded = tf.concat(values=[encoded, tf.reshape(l, -1)], axis=-1)
    return encoded


def flat_grad(grad):
    flatted = []
    for g in grad:
        flatted.append(tf.reshape(g, [-1]))
    return tf.concat(flatted, 0)

"""
True means x dominates y
"""
def dominance(x, y, weak = False):
    less_equal = tf.reduce_all(tf.math.less_equal(x,y),1)
    if weak:
        return less_equal
    less = tf.reduce_any(tf.math.less(x, y),1)
    return tf.logical_and(less_equal, less)

def normalize(arr, f, t=[-1, 1]):
    return t[0] + (arr - f[0]) * (t[1] - t[0]) / (f[1] - f[0] + 1e-8)
