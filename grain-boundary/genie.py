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
    print(trans_1)
    print(trans_2)
    box_1 = np.transpose(np.dot(trans_1, np.transpose(struct.coordinates)))
    box_2 = np.transpose(np.dot(trans_2, np.transpose(struct.coordinates)))
    search_size = 15
    coindicent_pts = find_coincident_points(box_1, box_2, search_size, 0.5)
    print(str(len(coindicent_pts)) + ' / ' + str(search_size**3 - 1))
    lattice = find_overlattice(coindicent_pts, PI / 6, PI / 2)
    print(lattice)
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
    return 0

def find_coincident_points(box_1, box_2, max_int, tol):
    search_points = cartesian_product(np.arange(max_int), 3)
    vecs = np.dot(search_points, box_1)
    nearest_int_mult = np.rint(np.dot(vecs, np.linalg.inv(box_2)))
    dist = np.apply_along_axis(np.linalg.norm, 1, (np.dot(nearest_int_mult, box_2) - vecs))
    vecs = vecs[np.where(dist <= tol)]
    vecs = vecs[np.argsort(np.apply_along_axis(np.linalg.norm, 1, vecs))]
    return vecs[1:]

def cartesian_product(array, level):
    res = []
    for i in range(level):
        res.append(np.tile(np.repeat(array, len(array) ** (level - i - 1)),
                           len(array) ** i))
    res = np.transpose(np.array(res))
    return res

def find_overlattice(coincident_pts, min_agl, max_agl, linear_eps=1e-5):
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
                    vol.append(det)
    res = np.array(res)
    vol = np.array(vol)
    vec_len_sum = np.array(map(lambda x : np.sum(x ** 2), res))
    res = res[np.argsort(vec_len_sum)]
    return res[0, :, :]

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

def apart_by_safe_distance(min_dist_dict, atom_1, atom_2):
    try:
        min_dist = min_dist_dict[(atom_1['element'], atom_2['element'])]
    except KeyError, e:
        try:
            min_dist = min_dist_dict[(atom_2['element'], atom_1['element'])]
        except KeyError, e:
            return True
        else:
            return np.linalg.norm(atom_1['position'] - 
                atom_2['position']) >= min_dist
    else:
        return np.linalg.norm(atom_1['position'] - 
                atom_2['position']) >= min_dist

def remove_collision_within_region(reg, min_dist_dict):
    i = 0
    while (i < len(reg) - 1):
        prev = reg[0:i+1]
        atm = reg[i]
        test = reg[i+1:]
        is_safe = np.array(map(lambda x : apart_by_safe_distance(
            min_dist_dict, x, atm), test))
        test = test[np.array(is_safe)]
        reg = np.concatenate((prev, test))
        i += 1
    return reg

def remove_collision_between_regions(reg_1, reg_2, min_dist_dict):
    for atm in reg_1:
        if len(reg_2) <= 0:
            break
        is_safe = np.array(map(lambda x : apart_by_safe_distance(
                               min_dist_dict, x, atm), reg_2))
        reg_2 = reg_2[is_safe]
    return reg_2

def remove_collision(struct, boundary_radius, min_dist_dict, 
                     random_delete=True):
    direct_length = abs(boundary_radius / np.linalg.norm(struct.coordinates[2]))
    print(direct_length)
    # First remove collision around the interface.
    # Select interface atoms and delete from original atom arrays.
    orig_atom_count = struct.cartesian.shape[0]
    print("Original atom #: " + str(orig_atom_count))
    on_iface_idx = np.logical_and(
        struct.direct['position'][:, 2] < (0.5 + direct_length),
        struct.direct['position'][:, 2] > (0.5 - direct_length))
    # Work in Cartesian system.
    iface_atoms = struct.cartesian[on_iface_idx]
    if random_delete:
        np.random.shuffle(iface_atoms)
    print(" Surface atom #: " + str(len(iface_atoms)))
    struct.cartesian = struct.cartesian[np.logical_not(on_iface_idx)]
    struct.direct = struct.direct[np.logical_not(on_iface_idx)]
    iface_atoms = remove_collision_within_region(iface_atoms, min_dist_dict)

    # Second remove collision by the boundary.
    on_btm_idx = np.logical_and(
        struct.direct['position'][:, 2] < (0.0 + direct_length),
        struct.direct['position'][:, 2] > (0.0 - direct_length))
    on_top_idx = np.logical_and(
        struct.direct['position'][:, 2] < (1.0 + direct_length),
        struct.direct['position'][:, 2] > (1.0 - direct_length))
    btm_atoms = struct.cartesian[on_btm_idx]
    if random_delete:
        np.random.shuffle(btm_atoms)
    top_atoms = struct.cartesian[on_top_idx]
    if random_delete:
        np.random.shuffle(top_atoms)
    print("Boundary atom #: " + str(len(btm_atoms) + len(top_atoms)))
    struct.cartesian = struct.cartesian[np.logical_not(
        np.logical_or(on_btm_idx, on_top_idx))]
    btm_atoms['position'] += struct.coordinates[2]

    # Remove atoms that are too close to each other within bottom or top slice.
    top_atoms = remove_collision_within_region(top_atoms, min_dist_dict)
    btm_atoms = remove_collision_within_region(btm_atoms, min_dist_dict)

    # Remove atom collisions 
    btm_atoms = remove_collision_between_regions(top_atoms, btm_atoms, 
                                                 min_dist_dict)
    btm_atoms['position'] -= struct.coordinates[2]

    struct.cartesian = np.concatenate((struct.cartesian, iface_atoms))
    struct.cartesian = np.concatenate((struct.cartesian, btm_atoms))
    struct.cartesian = np.concatenate((struct.cartesian, top_atoms))

    struct.reconcile(according_to='C')
    final_atom_count = struct.cartesian.shape[0]
    print("  Atoms removed: " + str(orig_atom_count - final_atom_count))

    return

def main(argv):
    struct = Structure.from_vasp(argv[1])
    gb_genie(struct, np.array([1., 1., 0.]), np.array([1., 0., 0.]), PI / 4, np.array([0, 0, 0]))

    # min_dist_dict = {
    #     ('Cd', 'Te') : 2.3, 
    #     ('Cd', 'Cd') : 2.6, 
    #     ('Te', 'Te') : 3.3
    # }
    # remove_collision(struct, 2.0, min_dist_dict, random_delete=False)
    # struct.to_xyz('col_rem_test')
    # struct.to_vasp('col_rem_test')

if __name__ == '__main__':
    main(sys.argv)