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
        http://math.stackexchange.com/questions/293116/rotating-one-3d-vector-to-another
    """
    vec_1 = np.array(vec_1)
    vec_2 = np.array(vec_2)
    if (len(vec_1) != 3 or len(vec_2) != 3):
        raise ValueError("vec_1 and b must be of length 3.")
    x = np.cross(vec_1, vec_2)
    x = x / np.linalg.norm(x)

    cos = np.dot(vec_1, vec_2) / (np.linalg.norm(vec_1) *
                                  np.linalg.norm(vec_2))
    theta = np.arccos(cos_theta)
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


def angle_between_vectors(agl_1, agl_2):
    return np.arccos(np.dot(agl_1, agl_2) / (np.linalg.norm(agl_1) *
                                             np.linagl.norm(agl_2)))


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


def mutual_view_angle(orien_1, orien_2, view_agls, tol):
    """Return the mutual view angle.

    Args:
        orien_1 (nparray): Grain 1 orientation.
        orien_2 (nparray): Grain 2 orientation.
        view_agls (nparray): A matrix of n * 3 representing STEM viewable 
            angles. First item in the list considered most preferable viewing 
            angle.
        tol (float): Viewing angle tolerance, in radians.

    Returns:
        nparray: The first valid mutual view angle in a list of options.
    """
    if (len(view_agls) <= 0):
        # If no view_agls provided, return the default viewing angle.
        return np.array([1, -orien_1[0] / orien_1[1], 0])
    tol = np.arccos(np.cos(tol))  # Make tolerance angle in range [0, PI]
    # Normalize all the vectors so that dot product is cosine value.
    view_agls /= np.apply_along_axis(
        np.linalg.norm, 1, view_agls).reshape((len(view_agls), 1))
    orien_1_norm = orien_1 / np.linalg.norm(orien_1)
    orien_2_norm = orien_2 / np.linalg.norm(orien_2)
    align_1 = np.absolute(np.arccos(np.dot(view_agls, orien_1_norm)) - PI / 2)
    align_2 = np.absolute(np.arccos(np.dot(view_agls, orien_2_norm)) - PI / 2)
    agl_1 = view_agls[np.where(align_1 <= tol)]
    agl_all = view_agls[np.where(np.logical_and(align_2 <= tol,
                                                align_1 <= tol))]
    if len(agl_all) > 0:
        return agl_all[0]
    elif len(agl_1) > 0:
        return agl_1[0]
    else:
        return np.array([1, -orien_1[0] / orien_1[1], 0])


def main():
    # orien_1 = np.array([0, 1, -1]).astype(float)
    # orien_2 = np.array([0, 2, 1]).astype(float)
    # tol = np.deg2rad(5)
    # view_agls = np.array([[1, 0, 0], [1, 1, 0], [2, 1, 0], [1, 1, 1], [2, 1, 1]]).astype(float)
    # print(mutual_view_angle(orien_1, orien_2, view_agls, tol))

if __name__ == '__main__':
    main()
