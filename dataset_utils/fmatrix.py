import numpy as np


def _cross_matrix(a):
    """
    Calculates cross product matrix for vector a.

    see Wikipedia: https://en.wikipedia.org/wiki/Cross_product#Conversion_to_matrix_multiplication
    :param a: 3-vector
    :return: 3x3 cross product matrix
    """
    result = np.zeros((3,3))
    for i in range(3):
        e_i_hat = np.zeros(3)
        e_i_hat[i] = -1
        result[i] = np.cross(a, e_i_hat)
    return result


def _calc_R(R1, R2):
    return R1.T.dot(R2)


def _calc_t(t1, t2):
    return t2 - t1


def essential_matrix_fom_Rt(R1, t1, R2, t2):
    """
    Compute essential matrix from R and t of both cameras.

    :param R1: rotation matrix of camera 1. np.array of dimension (3, 3)
    :param t1: translation vector of camera 1. np.array of dimension (3)
    :param R2: rotation matrix of camera 2. np.array of dimension (3, 3)
    :param t2: translation vector of camera 2. np.array of dimension (3)
    :return: Essential matrix. np.array of dimension (3, 3).
    """
    R = _calc_R(R1, R2)
    t = _calc_t(t1, t2)
    t_cross = _cross_matrix(t)

    return R @ t_cross


def fundamental_matrix_from_essential_matrix(K1, K2, E):
    """
    Computes f matrix from essential matrix and both camera intrinsics.

    :param K1: Camera intrinsics of camera 1. np.array of dimension (3, 3)
    :param K2: Camera intrinsics of camera 2. np.array of dimension (3, 3)
    :param E: Essential matrix. np.array of dimension (3, 3).
    :return: Fundamental matrix. np.array of dimension (3, 3)
    """
    K1_inv = np.linalg.inv(K1)
    K2_inv = np.linalg.inv(K2)
    return K2_inv.T @ E @ K1_inv


def camera_to_world(R, t):
    R_c = R.T
    t_c = -R.T @ t
    return R_c, t_c


def fundamental_matrix_from_camera_alternative(K1, R1, t1, K2, R2, t2, camera_coordinates=False):
    """
    Alternative way to calculate the fundamental matrix from camera params.

    This is not used any more, fundamental_matrix_from_camera is used instead.
    :param K1: camera intrinsics of camera 1. np.array of dimension (3, 3)
    :param R1: rotation matrix of camera 1. np.array of dimension (3, 3)
    :param t1: translation vector of camera 1. np.array of dimension (3)
    :param K2: camera intrinsics of camera 2
    :param R2: rotation matrix of camera 2. np.array of dimension (3, 3)
    :param t2: translation vector of camera 2. np.array of dimension (3)
    :param camera_coordinates: True if the camera parameters should be converted from
        camera coordinates into world coordinates first.
    :return: fundamental matrix. np.array of dimension (3, 3)
    """
    if camera_coordinates:
        # R and t where given in camera coordinates => compute world coordinates
        R1, t1, = camera_to_world(R1, t1)
        R2, t2, = camera_to_world(R2, t2)

    E = essential_matrix_fom_Rt(R1, t1, R2, t2)
    return fundamental_matrix_from_essential_matrix(K1, K2, E)


def camera_params_to_P(K, R, t):
    """
    Calcls camera projection matrix from camera intrinsics K, rotation R and translation t
    :param K: camera intrinsics. np.array of dimension (3, 3)
    :param R: rotation matrix. np.array of dimension (3, 3)
    :param t: translation vector. np.array of dimension (3)
    :return: P - projection matrix. np.array of dimension (3, 4)
    """
    Rt = np.zeros((4, 4))
    Rt[0:3, 0:3] = R
    Rt[0:3, 3] = t
    Rt[3, 3] = 1
    P = K @ \
        np.array([[1., 0., 0., 0.],
                  [0., 1., 0., 0.],
                  [0., 0., 1., 0.]]) \
        @ Rt
    return P


def fundamental_matrix_from_camera(K1, R1, t1, K2, R2, t2):
    """
    Computes fundamental matrix from the given camera parameters of camera 1 and 2.

    :param K1: camera intrinsics of camera 1. np.array of dimension (3, 3)
    :param R1: rotation matrix of camera 1. np.array of dimension (3, 3)
    :param t1: translation vector of camera 1. np.array of dimension (3)
    :param K2: camera intrinsics of camera 2
    :param R2: rotation matrix of camera 2. np.array of dimension (3, 3)
    :param t2: translation vector of camera 2. np.array of dimension (3)
    :return: fundamental matrix. np.array of dimension (3, 3)
    """
    # based on http://www.robots.ox.ac.uk/~vgg/hzbook/code/vgg_multiview/vgg_F_from_P.m
    P1 = camera_params_to_P(K1, R1, t1)
    P2 = camera_params_to_P(K2, R2, t2)

    X1 = P1[[1, 2], :]
    X2 = P1[[2, 0], :]
    X3 = P1[[0, 1], :]
    Y1 = P2[[1, 2], :]
    Y2 = P2[[2, 0], :]
    Y3 = P2[[0, 1], :]

    det_X1_Y1 = np.linalg.det(np.vstack((X1, Y1)))
    det_X2_Y1 = np.linalg.det(np.vstack((X2, Y1)))
    det_X3_Y1 = np.linalg.det(np.vstack((X3, Y1)))
    det_X1_Y2 = np.linalg.det(np.vstack((X1, Y2)))
    det_X2_Y2 = np.linalg.det(np.vstack((X2, Y2)))
    det_X3_Y2 = np.linalg.det(np.vstack((X3, Y2)))
    det_X1_Y3 = np.linalg.det(np.vstack((X1, Y3)))
    det_X2_Y3 = np.linalg.det(np.vstack((X2, Y3)))
    det_X3_Y3 = np.linalg.det(np.vstack((X3, Y3)))

    F = np.array([[det_X1_Y1, det_X2_Y1, det_X3_Y1],
                  [det_X1_Y2, det_X2_Y2, det_X3_Y2],
                  [det_X1_Y3, det_X2_Y3, det_X3_Y3]])

    return F


all_norms = [None, 'fro', 'abs']


def normalize(F, norm):
    """
    Normalizes the matrix F using the given norm.

    :param F: matrix to normalize. np.array of dimension (3, 3)
    :param norm: norm type. 'fro' for frobenius norm or 'abs' for maximum absolute value norm (inf norm)
    :return: normalized F. np.array of dimension (3, 3)
    """
    if norm is None:
        return F
    elif norm == 'abs':
        ord = np.inf
    elif norm == 'fro':
        ord = 'fro'
    else:
        raise ValueError('Unkown norm:' + norm + ', valid values are abs and fro')
    norm_value = np.linalg.norm(F, ord=ord)
    F_normed = F / norm_value
    return F_normed
