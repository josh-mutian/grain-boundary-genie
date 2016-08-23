import sys
import copy
import numpy as np
from structure import Structure
import geometry as geom
import collision_removal as coll_rm
from math import pi as PI


def gb_genie(struct, orien_1, orien_2, twist_agl, trans_vec):
    trans_1 = np.dot(geom.rotation_angle_matrix(np.array([0., 0., 1.]), 
                                                twist_agl),
                     geom.get_rotation_matrix(orien_1, np.array([0., 0., 1.])))
    trans_2 = geom.get_rotation_matrix(orien_2, np.array([0., 0., 1.]))
    print(trans_1)
    print(trans_2)
    box_1 = np.transpose(np.dot(trans_1, np.transpose(struct.coordinates)))
    box_2 = np.transpose(np.dot(trans_2, np.transpose(struct.coordinates)))
    search_size = 20
    coincident_pts = find_coincident_points(box_1, box_2, search_size, 0.5)
    print(str(len(coincident_pts)) + ' / ' + str(search_size**3 - 1))
    min_atom = 700
    max_atom = 10000
    min_vol = 0.5 * min_atom / (len(struct.direct) / np.linalg.det(struct.coordinates))
    max_vol = 0.5 * max_atom / (len(struct.direct) / np.linalg.det(struct.coordinates))
    lattice = find_overlattice(coincident_pts, PI / 4, PI / 2, min_vol, max_vol)
    print(lattice)
    lattice = lattice[0]
    print('Expected atoms: ' + str(2 * len(struct.direct) / np.linalg.det(struct.coordinates) * np.linalg.det(lattice)))
    struct_1 = struct
    struct_2 = copy.deepcopy(struct)
    struct_1.transform(trans_1)
    struct_2.transform(trans_2)
    print("Trans 1 det:" + str(np.linalg.det(trans_1)))
    print("Trans 2 det:" + str(np.linalg.det(trans_2)))
    struct_1.grow_to_supercell(lattice, 10000)
    struct_2.grow_to_supercell(lattice, 10000)
    struct_1.to_vasp('grown_1')
    struct_2.to_vasp('grown_2')
    glued_struct = combine_structures(struct_1, struct_2)
    glued_struct.to_vasp('grow_and_glue')
    glued_struct.to_xyz('grow_and_glue')

    min_dist_dict = {
        ('Cd', 'Te') : 3.0, 
        ('Cd', 'Cd') : 3.5, 
        ('Te', 'Te') : 3.3
    }
    coll_rm.remove_collision(glued_struct, 0.01, min_dist_dict, random_delete=False)
    glued_struct.to_xyz('col_rem_test')
    glued_struct.to_vasp('col_rem_test')

    return 0

def find_coincident_points(box_1, box_2, max_int, tol):
    search_points = geom.cartesian_product(np.arange(max_int), 3)
    vecs = np.dot(search_points, box_1)
    nearest_int_mult = np.rint(np.dot(vecs, np.linalg.inv(box_2)))
    dist = np.apply_along_axis(np.linalg.norm, 1, (np.dot(nearest_int_mult, box_2) - vecs))
    vecs = vecs[np.where(dist <= tol)]
    vecs = vecs[np.argsort(np.apply_along_axis(np.linalg.norm, 1, vecs))]
    return vecs[1:]

def find_overlattice(coincident_pts, min_agl, max_agl, min_vol, max_vol, 
                     max_pts=100, linear_eps=1e-5):
    print(min_vol)
    print(max_vol)
    if len(coincident_pts) < 3:
        raise ValueError('Must have at least 3 coincident points')
    if len(coincident_pts) > max_pts:
        print('Too many coincident points; reduced to %d.' % max_pts)
        coincident_pts = coincident_pts[0:max_pts]

    res = [] # Resulting lattice vectors: list of 3*3 nparrays.
    vol = [] # Volume of boxes: list of floats.
    for i in range(len(coincident_pts)):
        for j in range(i + 1, len(coincident_pts)):
            for k in range(j + 1, len(coincident_pts)):
                lat_vecs = coincident_pts[[i, j, k]]
                det = np.linalg.det(lat_vecs)
                if det < linear_eps:
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
    # res = res[good_vol_idx]
    # vol = vol[good_vol_idx]

    print('%d good lattice vector sets found.' % len(res))
    vec_len_sum = np.array(map(lambda x : np.sum(x ** 2), res))
    res = res[np.argsort(vec_len_sum)]
    return res

def combine_structures(struct_1, struct_2):
    # Assuming that these structures have same lattice vector sets. We extend
    # the c direction by 2 and shift struct_2 to that place. 
    struct_1.direct['position'][:, 2] /= 2.0
    struct_2.direct['position'][:, 2] /= 2.0
    struct_2.direct['position'][:, 2] += 0.5
    struct_1.to_vasp('test_1')
    struct_2.to_vasp('test_2')
    struct_1.direct = np.concatenate((struct_1.direct, struct_2.direct))
    struct_1.coordinates[2] *= 2.0
    struct_1.comment = struct_1.comment + '_' + struct_2.comment
    struct_1.reconcile(according_to='D')
    return struct_1

def main(argv):
    struct = Structure.from_vasp(argv[1])
    gb_genie(struct, np.array([1., 1., 0.]), np.array([1., 0., 0.]), PI / 4, np.array([0, 0, 0]))

    # min_dist_dict = {
    #     ('Cd', 'Te') : 3.0, 
    #     ('Cd', 'Cd') : 3.5, 
    #     ('Te', 'Te') : 3.3
    # }
    # coll_rm.remove_collision(struct, 0.01, min_dist_dict, random_delete=False)
    # struct.to_xyz('col_rem_test')
    # struct.to_vasp('col_rem_test')
    
if __name__ == '__main__':
    main(sys.argv)