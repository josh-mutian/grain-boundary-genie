import sys
import copy
import numpy as np
from structure import Structure
import geometry as geom
from math import pi as PI


def gb_genie(struct, orien_1, orien_2, twist_agl, trans_vec):
    trans_1 = np.dot(geom.rotation_angle_matrix(np.array([0., 0., 1.]), 
                                                twist_agl),
                     geom.get_rotation_matrix(orien_1, np.array([0., 0., 1.])))
    trans_2 = geom.get_rotation_matrix(orien_2, np.array([0., 0., 1.]))
    box_1 = np.transpose(np.dot(trans_1, np.transpose(struct.coordinates)))
    box_2 = np.transpose(np.dot(trans_2, np.transpose(struct.coordinates)))
    search_size = 20
    coindicent_pts = find_coincident_points(box_1, box_2, search_size, 1.0)
    # print(coindicent_pts)
    print(str(len(coindicent_pts)) + ' / ' + str(search_size**3))
    lattice = find_overlattice(coindicent_pts, PI / 6, PI / 2)
    print(lattice)
    struct.grow_to_supercell(lattice, 10000)
    struct.to_vasp('grow_1')
    return 0

def find_coincident_points(box_1, box_2, max_int, tol):
    # CAN BE VECTORIZED.
    res = []
    for i in range(max_int):
        for j in range(max_int):
            for k in range(max_int):
                vec = np.dot(np.array([i, j, k]), box_1)
                nearest_int_mult = np.rint(np.dot(vec, np.linalg.inv(box_2)))
                dist = np.linalg.norm(np.dot(nearest_int_mult, box_2) - vec)
                if dist <= tol:
                    res.append(vec)
    res = np.array(res)
    res = res[np.argsort(np.apply_along_axis(np.linalg.norm, 1, res))]
    return res[1:]

def cartesian_product(array, level, unique=False):
    res = np.zeros(len(array) ** level)
    for i in range(level):
        res = np.vstack((res, np.tile(np.repeat(array, 
                                                len(array) ** (level - i - 1)),
                                      len(array) ** i)))
    res = res[1:, :]
    return res

def find_overlattice(coincident_pts, min_agl, max_agl, linear_eps=1e-5):
    res = []
    for i in range(len(coincident_pts)):
        for j in range(i + 1, len(coincident_pts)):
            for k in range(j + 1, len(coincident_pts)):
                lat_vecs = coincident_pts[[i, j, k]]
                if abs(np.linalg.det(lat_vecs)) < linear_eps:
                    continue
                vec_agls = np.array([
                    geom.angle_between_vectors(lat_vecs[0], lat_vecs[1]),
                    geom.angle_between_vectors(lat_vecs[1], lat_vecs[2]), 
                    geom.angle_between_vectors(lat_vecs[2], lat_vecs[0])
                ])
                if np.all(vec_agls > min_agl) and np.all(vec_agls < max_agl):
                    res.append(lat_vecs)
    # lat_len_diff = np.array(map(
    #     lambda x : np.sum((x - np.apply_along_axis(np.mean, 0, x)) ** 2), 
    #     res))
    lat_len_diff = np.array(map(
        lambda x : np.sum(x ** 2), 
        res))
    res = np.array(res)
    print(res)
    res = res[np.argsort(lat_len_diff)]
    return res[0, :, :]


def main(argv):
    struct = Structure.from_vasp(argv[1])
    gb_genie(struct, np.array([1., 1., .0]), np.array([1., 2., 0]), 0, np.array([0, 0, 0]))
    
if __name__ == '__main__':
    main(sys.argv)