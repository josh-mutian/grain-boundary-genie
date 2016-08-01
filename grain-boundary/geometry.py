import numpy as np


def get_rotation_matrix(a, b):
    """Gets a rotation matrix from one vector to another vector.

    Args:
        a (nparray): An nparray of length 3 to represent the from vector.
        b (nparray): An nparray of length 3 to represent the from vector.

    Returns:
        nparray: A 3*3 array representing the rotation matrix.

    Raises:
        ValueError: Raised when arguments a or b has length other than 3.

    Reference:
        http://math.stackexchange.com/questions/293116/rotating-one-3d-vector-to-another
    """

    a = np.array(a)
    b = np.array(b)
    if (len(a) != 3 or len(b) != 3):
        raise ValueError("a and b must be of length 3.")
    x = np.cross(a, b)
    x = x / np.linalg.norm(x)

    cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    theta = np.arccos(cos_theta)
    A = np.array([
        [    0, -x[2],  x[1]],
        [ x[2],     0, -x[0]],
        [-x[1],  x[0],     0]
    ])  # skew-symmetric cross-product of v
    r = np.identity(3) + np.sin(theta) * A + (1 - cos_theta) * np.dot(A, A)
    return r

def rotation_angle_matrix(axis, agl):
    """Get the matrix of rotation about an axis.
    
    Args:
        axis (nparray): nparray of length 3.
        agl (float): Angle in degrees.
    
    Returns:
        nparray: nparray of dimension 3*3.
    """
    norm = np.linalg.norm(np.array(axis))
    axis = np.linalg.norm(np.array(axis))
    rad = agl / 360.0
    if rad == 0 or norm == 0:
        return np.identity(3)
    else:
        axis = axis / norm
    tensor_prod = np.dot(axis.reshape(3, 1), axis.reshape(3, 1))
    cross_prod = np.array([
        [       0, -axis[2],  axis[1]],
        [ axis[2],        0, -axis[0]],
        [-axis[1],  axis[0],        0]
    ])
    cos = np.cos(rad)
    sin = np.sin(rad)
    R = cos * np.identity(3) + sin * cross_prod + (1 - cos) * tensor_prod
    return R


def hausdorff_distance(point_set_1, point_set_2, default_value=10):
    """Calculates the Hausdorff distance between two point sets.

    Args:
        point_set_1 (nparray): An nparray of vectors (length 3).
        point_set_2 (nparray): An nparray of vectors (length 3).
        default_value (int, optional): The return value when either set 
            contains 0 points. Should be deprecated for the future when 
            better solution is found.

    Returns:
        float: The Hausdorff distance.
    """
    # point_set_1 and point_set_2 should have dimension of (_, 3) in Cartesian
    # and should both be numpy array
    n = point_set_1.shape[0]
    m = point_set_2.shape[0]

    if n == 0 or m == 0:
        return default_value

    # Reshape so that we can broadcast operations.
    pt_1 = np.reshape(point_set_1, (n, 1, 3))
    pt_2 = np.reshape(point_set_2, (m, 1, 3))

    # Euclidean distance is used here.
    # Calculate sup_x inf_y d(x, y)
    dist = np.apply_along_axis(
        np.linalg.norm, 2,
        np.repeat(pt_2, n, axis=1) - np.reshape(pt_1, (1, n, 3)))
    d_1 = np.amax(np.apply_along_axis(np.amin, 1, dist))
    # Calculate sup_y inf_x d(x, y)
    dist = np.apply_along_axis(
        np.linalg.norm, 2,
        np.repeat(pt_1, m, axis=1) - np.reshape(pt_2, (1, m, 3)))
    d_2 = np.amax(np.apply_along_axis(np.amin, 1, dist))

    return max(d_1, d_2)


def slice_distances(slice_1, slice_2):
    """Calculates the weighted Hausdorff distance between two slices.

    Args:
        slice_1 (nparray list): A list of nparray (each row representing a 3-D
            vector), each nparray represents positions of atoms of one type 
            of element.
        slice_2 (nparray list): (Ditto.)

    Returns:
        float: The Hausdorff distance weighted by numbers of each element.

    Raises:
        ValueError: Raised when slices have different numbers of elements.
            Should be deprecated for the future when better solution is found.
    """
    # slice: list of nparrays, each containing a bunch of length 3D coord.
    if len(slice_1) != len(slice_2):
        raise ValueError('Two slices must have same kinds of elements')

    element_count = (np.array(map(len, slice_1)) + np.array(map(len, slice_2)))
    element_frac = element_count / float(np.sum(element_count))

    hausdorff_dists = np.array([hausdorff_distance(x, y) for (x, y) in
                                zip(slice_1, slice_2)])

    return np.dot(element_frac, hausdorff_dists)
