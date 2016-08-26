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
        min_agl (float): The minimum angle allowed between any vector and the 
            plane formed by the other two vectors, in rad.
        max_agl (float): The maximum angle allowed between any vector and the 
            plane formed by the other two vectors, in rad.
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
        print('Too many coincident points: %d reduced to %d.' % \
            (len(coincident_pts), max_pts))
        coincident_pts = coincident_pts[np.argsort(
            np.apply_along_axis(np.linalg.norm, 1, coincident_pts))]
        coincident_pts = coincident_pts[0:max_pts]

    print('Processing %d coincidence points.' % len(coincident_pts))
    res = []  # Resulting lattice vectors: list of 3*3 nparrays.
    for i in range(len(coincident_pts)):
        for j in range(i + 1, len(coincident_pts)):
            for k in range(j + 1, len(coincident_pts)):
                lat_vecs = coincident_pts[[i, j, k]]
                res.append(lat_vecs)
    print('%d candidate lattice vector sets.' % len(res))
    res = np.array(res)
    # Check volume criterion.
    vol = np.absolute(np.linalg.det(res))
    res = res[np.logical_and(vol < max_vol, vol > min_vol)]
    # Check vector lengths.
    shortest_vec_len = np.apply_along_axis(np.linalg.norm, 2, res)
    res = res[np.apply_along_axis(
        np.all, 1, shortest_vec_len > min_vec_len)]
    # Check angles.
    vec_agls = np.array(map(geom.get_box_angles, res.tolist()))
    res = res[np.apply_along_axis(
        np.all, 1, np.logical_and(vec_agls > min_agl, vec_agls < max_agl))]
    if len(res) <= 0:
        raise ValueError('No lattice vector set that meets requirements.')
    print('Totally %d qualified lattice vector sets found' % len(res))
    return res[np.argsort(np.absolute(np.linalg.det(res)))]
