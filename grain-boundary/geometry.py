import numpy as np


def get_rotation_matrix(a, b):
    '''
    a, b are vectors represented by a 1*3 arrays.
    a -> b
    ref: http://math.stackexchange.com/questions/293116/rotating-one-3d-vector-to-another
    '''
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


def hausdorff_distance(point_set_1, point_set_2, default_value=10):
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
    # slice: list of nparrays, each containing a bunch of length 3D coord.
    if len(slice_1) != len(slice_2):
        raise ValueError('Two slices must have same kinds of elements')

    element_count = (np.array(map(len, slice_1)) + np.array(map(len, slice_2)))
    element_frac = element_count / float(np.sum(element_count))

    hausdorff_dists = np.array([hausdorff_distance(x, y) for (x, y) in
                                zip(slice_1, slice_2)])

    return np.dot(element_frac, hausdorff_dists)
