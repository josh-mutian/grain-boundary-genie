"""Functions used to search for coincidence points and lattice vectors.
"""
import numpy as np
import geometry as geom


def find_coincidence_points(box_1, box_2, max_int, tol):
    """Searches for coincidence points.

    Args:
        box_1 (nparray): Coordinate system of a structure (3 * 3).
        box_2 (nparray): COordinate system of another structure (3 * 3).
        max_int (int): Maximum integer to grow and search.
        tol (float): Tolerance of distance between coincidence points, in 
            angstrom.

    Returns:
        nparray: A matrix of coincidence points (n * 3).
    """
    search_points = geom.cartesian_product(np.arange(max_int), 3)
    vecs = np.dot(search_points, box_1)
    nearest_int_mult = np.rint(np.dot(vecs, np.linalg.inv(box_2)))
    dist = np.apply_along_axis(
        np.linalg.norm, 1, (np.dot(nearest_int_mult, box_2) - vecs))
    vecs = vecs[np.where(dist <= tol)]
    vecs = vecs[np.argsort(np.apply_along_axis(np.linalg.norm, 1, vecs))]
    return vecs[1:]


def find_overlattice(coincident_pts, min_agl, max_agl, min_vol, max_vol,
                     max_pts=100, min_vec_len=0.):
    """Searches for sets of three coincidence points and output the sets that
    meets all the requirements.

    Args:
        coincident_pts (nparray): A matrix of coincidence points (n * 3).
        min_agl (float): The minimum angle allowed between any two vectors, 
            in rad.
        max_agl (float): The maximum angle allowed between any two vectors, 
            in rad.
        min_vol (float): The minimum angle allowed between any two vectors, 
            in angstrom.
        max_vol (float): The maximum angle allowed between any two vectors, 
            in angstrom.
        max_pts (int, optional): Maximum number of coincidence points to 
            search for the set of three vectors.
        min_vec_len (float, optional): The minimum length of any vector,
            in angstrom.

    Returns:
        nparray: An nparray consisting a list of 3-vector sets (n * 3 * 3).

    Raises:
        ValueError: Raised when the input coincidence point list has length
            less than 3 or no result can meet all requirements.
    """
    if len(coincident_pts) < 3:
        raise ValueError('Must have at least 3 coincident points')
    if len(coincident_pts) > max_pts:
        print('Too many coincident points; reduced to %d.' % max_pts)
        coincident_pts = coincident_pts[0:max_pts]

    res = []  # Resulting lattice vectors: list of 3*3 nparrays.
    vol = []  # Volume of boxes: list of floats.
    for i in range(len(coincident_pts)):
        for j in range(i + 1, len(coincident_pts)):
            for k in range(j + 1, len(coincident_pts)):
                lat_vecs = coincident_pts[[i, j, k]]
                det = np.linalg.det(lat_vecs)
                if det < 1e-5:
                    continue
                vec_lens = np.apply_along_axis(np.linalg.norm, 1, lat_vecs)
                if not np.all(vec_lens > min_vec_len):
                    continue
                vec_agls = np.array([
                    geom.angle_between_vectors(lat_vecs[0], lat_vecs[1]),
                    geom.angle_between_vectors(lat_vecs[1], lat_vecs[2]),
                    geom.angle_between_vectors(lat_vecs[2], lat_vecs[0])
                ])
                if np.all(vec_agls > min_agl) and np.all(vec_agls < max_agl):
                    res.append(lat_vecs)
                    vol.append(abs(det))
    res = np.array(res)
    vol = np.array(vol)
    good_vol_idx = np.logical_and(vol < max_vol, vol > min_vol)
    res = res[good_vol_idx]
    vol = vol[good_vol_idx]

    print('%d good lattice vector sets found.' % len(res))
    res = res[np.argsort(vol)]
    if len(res) <= 0:
        raise ValueError('No lattice vector set meeting all requirements.')
    else:
        return res
