import tensorflow as tf
from tensorflow.python.keras.layers import Lambda, Dense, Reshape
import numpy as np


def get_K_inv(f_inv):
    """
    Creates the inverse of the intrinsic matrix K(-1) from the inverse focal length f(-1).

    Only one intrinsic parameter is used, the focal length f = alpha_x = alpha_y.
        | f | 0 | 0 |
    K = | 0 | f | 0 |
        | 0 | 0 | 1 |

            | 1 / f | 0     | 0 |
    K(-1) = | 0     | 1 / f | 0 |
            | 0     | 0     | 1 |

    :param f_inv: inverse of focal length (f(-1)). Tensor (tf) of dimension (None, 1).
    :return: K_inv, inverse of intrinsic matrix (K(-1)). Tensor (tf) of dimension (None, 3, 3).
    """
    f_inv = tf.reshape(f_inv, [-1, 1])
    one = tf.ones_like(f_inv, dtype=tf.float32)
    zero = tf.zeros_like(f_inv, dtype=tf.float32)
    K_inv = tf.stack([tf.concat([f_inv, zero, zero], axis=1),
                      tf.concat([zero, f_inv, zero], axis=1),
                      tf.concat([zero, zero, one], axis=1)], axis=1)
    return K_inv


def get_K_inv_ext(alpha_x_inv, alpha_y_inv, u_0, v_0):
    """
    Creates the inverse of the intrinsic matrix K(-1) from the inverse focal lengths alpha_x(-1), alpha_y(-1) and
    the princiap point u_0, v_0.

    The 4 intrinsic parameters alpha_x, alpha_y, u_0 and v_0 are used (alpha_x = f*m_x, alpha_y = f*m_y).
         | alpha_x | 0       | u_0 |
     K = | 0       | alpha_y | v_0 |
         | 0       | 0       | 1   |

             | 1 / alpha_x | 0           | - u_0 / alpha_x |
     K(-1) = | 0           | 1 / alpha_y | - v_0 / alpha_y |
             | 0           | 0           | 1               |

    :param alpha_x_inv: inverse of x focal length (alpha_x(-1)). Tensor (tf) of dimension (None, 1).
    :param alpha_y_inv: inverse of y focal length (alpha_y(-1)). Tensor (tf) of dimension (None, 1).
    :param u_0: x coordinate of principal point. Tensor (tf) of dimension (None, 1).
    :param v_0: y coordinate of principal point. Tensor (tf) of dimension (None, 1).
    :return: K_inv, inverse of intrinsic matrix (K(-1)). Tensor (tf) of dimension (None, 3, 3).
    """
    alpha_x_inv = tf.reshape(alpha_x_inv, [-1, 1])
    alpha_y_inv = tf.reshape(alpha_y_inv, [-1, 1])
    u_0 = tf.reshape(u_0, [-1, 1])
    v_0 = tf.reshape(v_0, [-1, 1])
    one = tf.ones_like(alpha_x_inv, dtype=tf.float32)
    zero = tf.zeros_like(alpha_x_inv, dtype=tf.float32)

    K_inv = tf.stack([tf.concat([alpha_x_inv, zero,        -u_0*alpha_x_inv], axis=1),
                      tf.concat([zero,        alpha_y_inv, -v_0*alpha_y_inv], axis=1),
                      tf.concat([zero,        zero,        one             ], axis=1)], axis=1)
    return K_inv


def get_R(rx, ry, rz):
    """
    Gets 3-dimensional rotation matrix from the given rotation angles.

    :param rx: Rotation angle around x-axis in radians. Tensor (tf) of dimension (None, 1).
    :param ry: Rotation angle around y-axis in radians. Tensor (tf) of dimension (None, 1).
    :param rz: Rotation angle around z-axis in radians. Tensor (tf) of dimension (None, 1).
    :return: Rotation matrix. Tensor (tf) of dimension (None, 3, 3).
    """
    rx = tf.reshape(rx, [-1, 1])
    ry = tf.reshape(ry, [-1, 1])
    rz = tf.reshape(rz, [-1, 1])

    cos_rx = tf.cos(rx)
    sin_rx = tf.sin(rx)
    cos_ry = tf.cos(ry)
    sin_ry = tf.sin(ry)
    cos_rz = tf.cos(rz)
    sin_rz = tf.sin(rz)
    one = tf.ones_like(cos_rx, dtype=tf.float32)
    zero = tf.zeros_like(cos_rx, dtype=tf.float32)

    R_x = tf.stack([tf.concat([one, zero, zero], axis=1),
                    tf.concat([zero, cos_rx, sin_rx], axis=1),
                    tf.concat([zero, -sin_rx, cos_rx], axis=1)], axis=1)

    R_y = tf.stack([tf.concat([cos_ry, zero, -sin_ry], axis=1),
                    tf.concat([zero, one, zero], axis=1),
                    tf.concat([sin_ry, zero, cos_ry], axis=1)], axis=1)

    R_z = tf.stack([tf.concat([cos_rz, sin_rz, zero], axis=1),
                    tf.concat([-sin_rz, cos_rz, zero], axis=1),
                    tf.concat([zero, zero, one], axis=1)], axis=1)

    R = R_x @ R_y @ R_z
    return R


# t: (batch x 3)
def get_t_cross(tx, ty, tz):
    """
    Gets cross product matrix for 3 dimensional vector with components tx, ty, tz.

              | 0   | -tz | ty  |
    t_cross = | tz  | 0   | -tx |
              | -ty | tx  | 0   |

    :param tx: x component of vector. Tensor (tf) of dimension (None, 1).
    :param ty: y component of vector. Tensor (tf) of dimension (None, 1).
    :param tz: z component of vector. Tensor (tf) of dimension (None, 1).
    :return: Cross product matrix of t. Tensor (tf) of dimension (None, 3, 3).
    """
    tx = tf.reshape(tx, [-1, 1])
    ty = tf.reshape(ty, [-1, 1])
    tz = tf.reshape(tz, [-1, 1])
    zero = tf.zeros_like(tx, dtype=tf.float32)

    t_cross = tf.stack([tf.concat([zero, -tz, ty], axis=1),
                        tf.concat([tz, zero, -tx], axis=1),
                        tf.concat([-ty, tx, zero], axis=1)], axis=1)
    return t_cross


def reconstruction_fn(x):
    """
    Reconstructs the fundamental matrix from the 8 given params.

    F = K2_inv * t_cross * R * K1_inv

    :param x: Input vector containing the params f1_inv, f2_inv, rx, ry, rz, tx, ty, tz.
        Tensor (tf) of dimension (None, 8).
    :return: Fundamental matrix. Tensor (tf) of dimension (None, 3, 3).
    """
    f1 = x[:, 0]
    f2 = x[:, 1]
    rx = x[:, 2]
    ry = x[:, 3]
    rz = x[:, 4]
    tx = x[:, 5]
    ty = x[:, 6]
    tz = x[:, 7]

    K1_inv = get_K_inv(f1)
    K2_inv = get_K_inv(f2)
    t_cross = get_t_cross(tx, ty, tz)
    R = get_R(rx, ry, rz)

    F = K2_inv @ t_cross @ R @ K1_inv
    return F


# x: (batch x 14) tensor
# (alpha_x_1, alpha_y_1, u_0_1, v_0_1, alpha_x_2, alpha_y_2, u_0_2, v_0_2, rx, ry, rz, tx, ty, tz)
def reconstruction_fn_ext(x):
    """
    Reconstructs the fundamental matrix from the 14 given params. (uses extended intrinsic matrix)

    F = K2_inv * t_cross * R * K1_inv

    :param x:  Input vector containing the params alpha_x_1_inv, alpha_y_1_inv, u_0_1, v_0_1,
        alpha_x_2_inv, alpha_y_2_inv, u_0_2, v_0_2, rx, ry, rz, tx, ty, tz.
        Tensor (tf) of dimension (None, 14).
    :return: Fundamental matrix. Tensor (tf) of dimension (None, 3, 3).
    """
    alpha_x_1 = x[:, 0]
    alpha_y_1 = x[:, 1]
    u_0_1 = x[:, 2]
    v_0_1 = x[:, 3]
    alpha_x_2 = x[:, 4]
    alpha_y_2 = x[:, 5]
    u_0_2 = x[:, 6]
    v_0_2 = x[:, 7]
    rx = x[:, 8]
    ry = x[:, 9]
    rz = x[:, 10]
    tx = x[:, 11]
    ty = x[:, 12]
    tz = x[:, 13]

    K1_inv = get_K_inv_ext(alpha_x_1, alpha_y_1, u_0_1, v_0_1)
    K2_inv = get_K_inv_ext(alpha_x_2, alpha_y_2, u_0_2, v_0_2)
    t_cross = get_t_cross(tx, ty, tz)
    R = get_R(rx, ry, rz)

    F = K2_inv @ t_cross @ R @ K1_inv

    return F


def reconstruction_layer_simple_fn(x):
    """
    Applies the simple reconstruction layer that just uses normal regression and no special reconstruction.

    :param x: Input tensor of any flat dimension (None, input_neurons).
    :return: Reconstructed fundamental matrix. Tensor (tf) of dimension (None, 3, 3).
    """
    x = Dense(9, name='regressor_reg_fc3', activation=None)(x)
    x = Reshape((3, 3))(x)
    return x


def reconstruction_layer_reconstruct_fn(x):
    """
    Applies the reconstruction layer using the normal reconstruction function (with 8 params).

    :param x: Input tensor of any flat dimension (None, input_neurons).
    :return: Reconstructed fundamental matrix. Tensor (tf) of dimension (None, 3, 3).
    """
    x = Dense(8, name='regressor_reg_fc3', activation=None)(x)

    x = Lambda(reconstruction_fn, (3, 3), name='regressor_reg_reconstruct')(x)

    return x


def reconstruction_layer_reconstruct_ext_fn(x):
    """
    Applies the reconstruction layer using the extended reconstruction function (with 14 params).

    :param x: Input tensor of any flat dimension (None, input_neurons).
    :return: Reconstructed fundamental matrix. Tensor (tf) of dimension (None, 3, 3).
    """
    x = Dense(14, name='regressor_reg_fc3', activation=None)(x)
    x = Lambda(reconstruction_fn_ext, (3, 3), name='regressor_reg_reconstruct_ext')(x)
    return x


def reconstruction_layer(reconstruction):
    """
    Creates a reconstruction layer for the given reconstruction type.

    :param reconstruction: Type of reconstruction: None (for simple reconstruction), 'reconstruct' or 'reconstruct_ext'.
    :return: Reconstruction layer which accepts any flat input tensor (None, input_neurons) and outputs
        a fundamental matrix (None, 3, 3).
    """
    if reconstruction is None:
        return reconstruction_layer_simple_fn
    elif reconstruction == 'reconstruct':
        return reconstruction_layer_reconstruct_fn
    elif reconstruction == 'reconstruct_ext':
        return reconstruction_layer_reconstruct_ext_fn
    else:
        raise ValueError('Unknown reconstruction: %s' % reconstruction)


def norm_layer_fn(F, order):
    """
    Normalizes the given F matrix.

    :param F: Input matrix. Tensor (tf) of dimension (None, 3, 3).
    :param order: Type of norm. Either 'fro' or 'abs'.
    :return: Normed matrix. Tensor (tf) of dimension (None, 3, 3).
    """
    if order == 'fro':
        norm = tf.norm(F, ord='fro', axis=(1, 2))
    elif order == 'abs':
        norm = tf.norm(F, ord=np.inf, axis=(1, 2))
    else:
        raise ValueError('Unknown norm order: %s' % order)

    norm = tf.reshape(norm, [-1, 1, 1])
    normed = tf.multiply(F, tf.reciprocal(norm))

    return normed


def norm_layer(order):
    """
    Creates a norm layer for the given norm order.

    :param order: Type of norm. Either 'fro' or 'abs'.
    :return: Norm layer which accepts a tensor of dimensino (None, 3, 3)
        and returns the normed matrix of same dimension.
    """
    return Lambda(lambda x: norm_layer_fn(x, order), name=('regressor_reg_normalize_%s' % order))
