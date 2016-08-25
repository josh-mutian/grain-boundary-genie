import numpy as np
from math import pi as PI


def get_rotation_matrix(vec_1, vec_2):
    """Gets a rotation matrix from one vector to another vector.

    Args:
        vec_1 (nparray): An nparray of length 3 to represent the from vector.
        vec_2 (nparray): An nparray of length 3 to represent the from vector.

    Returns:
        nparray: A 3*3 array representing the rotation matrix.

    Raises:
        ValueError: Raised when arguments a or b has length other than 3.

    Reference:
        http://math.stackexchange.com/questions/293116/rotating-one-3d-vector-
            to-another
    """
    vec_1 = np.array(vec_1)
    vec_2 = np.array(vec_2)
    if (len(vec_1) != 3 or len(vec_2) != 3):
        raise ValueError("vec_1 and b must be of length 3.")
    x = np.cross(vec_1, vec_2)
    x = x / np.linalg.norm(x) if np.linalg.norm(x) != 0 else x

    cos = np.dot(vec_1, vec_2) / (np.linalg.norm(vec_1) *
                                  np.linalg.norm(vec_2))
    theta = np.arccos(cos)
    sin = np.sin(theta)
    cross_prod = np.array([
        [    0, -x[2],  x[1]],
        [ x[2],     0, -x[0]],
        [-x[1],  x[0],     0]
    ])  # skew-symmetric cross-product of v
    r = np.identity(3) + sin * cross_prod + \
        (1 - cos) * np.dot(cross_prod, cross_prod)
    return r


def rotation_angle_matrix(axis, agl):
    """Get the matrix of rotation about an axis.

    Args:
        axis (nparray): nparray of length 3.
        agl (float): Angle in radians.

    Returns:
        nparray: nparray of dimension 3*3.
    """
    norm = np.linalg.norm(np.array(axis))
    if agl == 0 or norm == 0:
        return np.identity(3)
    else:
        axis = axis / norm
    tensor_prod = np.dot(axis.reshape(3, 1), axis.reshape(1, 3))
    cross_prod = np.array([
        [       0, -axis[2],  axis[1]],
        [ axis[2],        0, -axis[0]],
        [-axis[1],  axis[0],        0]
    ])
    cos = np.cos(agl)
    sin = np.sin(agl)
    r = cos * np.identity(3) + sin * cross_prod + (1 - cos) * tensor_prod
    return r


def angle_between_vectors(vec_1, vec_2):
    """Calculate angle between vectors.

    Args:
        vec_1 (nparray): Vector (3).
        vec_2 (nparray): Vector (3). 

    Returns:
        float: Angle between vectors (in rad), ranging [0, PI).
    """
    return np.arccos(np.dot(vec_1, vec_2) /
                     (np.linalg.norm(vec_1) * np.linalg.norm(vec_2)))


def normalize_vector(vec):
    """Normalizes a vector.

    Args:
        vec (nparray): Vector (3).

    Returns:
        nparray: Normalized vector (3).
    """
    return vec / np.linalg.norm(vec)


def valid_direct_vec(vec):
    """Determines whether a vector is a valid vector in direct mode.

    Args:
        vec (nparray): Vector (3).

    Returns:
        boolean: Returns True if all three coordinates are in range [0., 1.]
            with a tolerance of 1e-5.
    """
    return np.all(np.absolute(vec - 0.5) < 0.5 + 1e-5)


def cartesian_product(array, level):
    """Get the Cartesian products of an array.

    Args:
        array (nparray): The array, must be 1-D.
        level (int): Folds of Cartesian Product.

    Returns:
        nparray: A nparray of Cartesian products (len(array) ^ level, level).
    """
    res = []
    for i in range(level):
        res.append(np.tile(np.repeat(array, len(array) ** (level - i - 1)),
                           len(array) ** i))
    res = np.transpose(np.array(res))
    return res
