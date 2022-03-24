import tensorflow as tf
import tensorflow_probability as tfp
from swarm import utils


def dimensions(layers):
    """Returns the number of `weights + biases` of a neural net, given the layers.

    Args:
        layers (list): List of 'int' representing the size of each layer.

    Returns:
        int: The number of optimizable variables (`weights + biases`)
    """
    sum = 0
    for l in range(0, len(layers) - 1):
        sum += layers[l] * layers[l + 1] + layers[l + 1]
    return sum


def progress(percent=0, width=30, metric=None, metricValue=None):
    """Prints on screen the current progress of a process.

    Args:
        percent (int, optional): The current progress (in percentiles). Defaults to 0.
        width (int, optional): The width size of the progress bar. Defaults to 30.
        metric (float, optional): The metric used. Defaults to None.
        metricValue (str, optional): The unit name for the metric used. Defaults to None.
    """
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


def encode(weights, biases):
    """Given weights and biases of a neural net, it returns a flat `tf.Tensor` with those values. This *encoded* format represents a particle in the PSO.

    Args:
        weights (tf.Tensor): The weights of the neural net.
        biases (tf.Tensor): The biases of the neural net.

    Returns:
        tf.Tensor: The particle for PSO. The flattened tensor with weights and biases.
    """
    encoded = tf.Variable([])
    for l in weights:
        encoded = tf.concat(values=[encoded, tf.reshape(l, -1)], axis=-1)
    for l in biases:
        encoded = tf.concat(values=[encoded, tf.reshape(l, -1)], axis=-1)
    return encoded


def decode(encoded, layers):
    """It will decode a PSO particle into the weights and biases of a neural network. It does the inverse process of the `encode` function.

    Args:
        encoded (tf.Tensor): The PSO particle.
        layers (list): List of 'int' representing the size of each layer.

    Returns:
        tuple: Two `tf.Tensor` representing the weights and biases of a neural net.
    """
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
    """It runs the multilayer perceptron neural network. Given the weights and biases representing the neural net and the input population `X`.

    Args:
        weights (tf.Tensor): The weights of the neural net.
        biases (tf.Tensor): The biases of the neural net.
        X (tf.Tensor): The input values.
        x_min (int, optional): The floor value for the normalization. Defaults to -1.
        x_max (int, optional): The roof value for the normalization. Defaults to 1.

    Returns:
        tf.Tensor: The prediction `Y`.
    """
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
    """Replace `nan` with zeros. **CAUTION**: `nan` may be the result of an infinitely small number, but it could happen the other way around too. If the `nan` was the result of an infinitely big number, the zero representation would be misleading.

    Args:
        t (tf.Tensor): The tensor with `nan` values.

    Returns:
        tf.Tensor: Tensor with `0s` instead of `nan`.
    """
    return tf.where(tf.math.is_nan(t), tf.zeros_like(t), t)


def layer_init(size, method):
    """Initialization for normalized a layer.

    Args:
        size (int): The layer size.

    Returns:
        tf.Tensor: The weights for the layer.
    """
    in_dim = size[0]
    out_dim = size[1]
    _stddev = tf.sqrt(6 / (in_dim + out_dim))
    if method == "he":
        _stddev = tf.sqrt(2 / (in_dim))
    if method == "lecun":
        _stddev = tf.sqrt(1 / (in_dim))
    return tf.Variable(
        tf.random.truncated_normal([in_dim, out_dim], stddev=_stddev),
        dtype=tf.float32,
    )


def initialize_NN(layers, method):
    """Initialize a neural network following the initialization given by `method`.

    Args:
        layers (list): A list of `int` representing each layer size.

    Returns:
        tuple: Two `tf.Tensor` with the weights and biases of the neural net.
    """
    weights = []
    biases = []
    num_layers = len(layers)
    for l in range(0, num_layers - 1):
        W = layer_init([layers[l], layers[l + 1]], method)
        b = tf.Variable(
            tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32
        )
        weights.append(W)
        biases.append(b)
    return weights, biases


def _build_normalized(pop_size, layer_sizes, method):
    """Initialize multiple neural networks using a normalized initialization selected by `method`.

    Args:
        pop_size (int): Number of neural networks to initialize.
        layers (list): A list of `int` representing each layer size. (All the neural nets must have the same topology).

    Returns:
        tf.Tensor: All the neural nets.
    """
    init_nns = []
    for _ in range(pop_size):
        w, b = initialize_NN(layer_sizes, method)
        new_nn = encode(w, b)
        init_nns.append(new_nn)
    return tf.Variable(init_nns, dtype=tf.float32)


def _build_uniform(pop_size, layer_sizes, _):
    """Initialize multiple neural networks using a uniformly distributed initialization.

    Args:
        pop_size (int): Number of neural networks to initialize.
        layers (list): A list of `int` representing each layer size. (All the neural nets must have the same topology).

    Returns:
        tf.Tensor: All the neural nets.
    """
    x_min = -1
    x_max = 1
    dim = dimensions(layer_sizes)
    return tf.Variable(tf.random.uniform([pop_size, dim], x_min, x_max))


def _build_logLogistic(pop_size, layer_sizes, _):
    """Initialize multiple neural networks using a log-logistic initialization.

    Args:
        pop_size (int): Number of neural networks to initialize.
        layers (list): A list of `int` representing each layer size. (All the neural nets must have the same topology).

    Returns:
        tf.Tensor: All the neural nets.
    """
    dim = dimensions(layer_sizes)
    dist = tfp.distributions.LogLogistic(0, 0.1)
    return dist.sample([pop_size, dim])


def build_NN(pop_size, layer_sizes, method):
    """Initialize multiple neural networks using a normalized initialization selected by `method`.

    Args:
        pop_size (int): Number of neural networks to initialize.
        layers (list): A list of `int` representing each layer size. (All the neural nets must have the same topology).

    Returns:
        tf.Tensor: All the neural nets.
    """
    methods = {
        "uniform": "_build_uniform",
        "log_logistic": "_build_logLogistic",
        "xavier": "_build_normalized",
        "he": "_build_normalized",
        "lecun": "_build_normalized",
    }
    return getattr(utils, methods.get(method, "_build_normalized"))(
        pop_size, layer_sizes, method
    )


def flat_grad(grad):
    """Flattens the gradient tensor.

    Args:
        grad (tf.Tensor): Gradients

    Returns:
        tf.Tensor: Flatted gradients.
    """
    flatted = []
    for g in grad:
        flatted.append(tf.reshape(g, [-1]))
    return tf.concat(flatted, 0)


def dominance(x, y, weak=False):
    """Dominance test. True means x dominates y.

    Args:
        x (tf.Tensor): A Tensor. Must be one of the following types: float32, float64.
        y (tf.Tensor): 	A Tensor. Must have the same type as x.
        weak (bool, optional): True for the weak dominance. Defaults to False.

    Returns:
        tf.Tensor: A Tensor of type bool.
    """
    less_equal = tf.reduce_all(tf.math.less_equal(x, y), 1)
    if weak:
        return less_equal
    less = tf.reduce_any(tf.math.less(x, y), 1)
    return tf.logical_and(less_equal, less)


def normalize(arr, t=[-1, 1]):
    """Min-Max normalization

    Args:
        arr (tf.Tensor): A Tensor. Must be one of the following types: float32, float64.
        t (list, optional): Range for the final tensor. Defaults to [-1, 1].

    Returns:
        tf.Tensor: A Tensor of same type as x.
    """
    max = tf.reduce_max(arr)
    min = tf.reduce_min(arr)
    return t[0] + (arr - min) * (t[1] - t[0]) / (max - min + 1e-8)
