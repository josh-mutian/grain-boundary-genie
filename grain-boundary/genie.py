import sys
import copy
import numpy as np
from structure import Structure
import geometry as geom
import collision_removal as coll_rmvl
import coincidence_search as coin_srch
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
    coincident_pts = coin_srch.find_coincident_points(box_1, box_2, search_size, 0.5)
    print(str(len(coincident_pts)) + ' / ' + str(search_size**3 - 1))
    min_atom = 0
    max_atom = 10000
    min_vol = 0.5 * min_atom / (len(struct.direct) / np.linalg.det(struct.coordinates))
    max_vol = 0.5 * max_atom / (len(struct.direct) / np.linalg.det(struct.coordinates))
    lattice = coin_srch.find_overlattice(coincident_pts, PI / 4, PI / 2, min_vol, max_vol, min_vec_len=0.0)
    print(lattice)
    lattice = lattice[0]
    print('Expected atoms: ' + str(2 * len(struct.direct) / np.linalg.det(struct.coordinates) * np.linalg.det(lattice)))
    struct_1 = struct
    struct_2 = copy.deepcopy(struct)
    struct_1.transform(trans_1)
    struct_2.transform(trans_2)
    struct_1.grow_to_supercell(lattice, 10000)
    struct_2.grow_to_supercell(lattice, 10000)
    struct_1.to_vasp('grown_1')
    struct_2.to_vasp('grown_2')
    glued_struct = Structure.combine_structures(struct_1, struct_2)
    glued_struct.to_vasp('grow_and_glue')
    glued_struct.to_xyz('grow_and_glue')

    min_dist_dict = {
        ('Cd', 'Te') : 2.8, 
        ('Cd', 'Cd') : 3.0, 
        ('Te', 'Te') : 3.0
    }
    coll_rmvl.remove_collision(glued_struct, 0.025, min_dist_dict, random_delete=False)
    glued_struct.to_xyz('col_rem_test')
    glued_struct.to_vasp('col_rem_test')

    return 0


def main(argv):
    struct = Structure.from_vasp(argv[1])
    gb_genie(struct, np.array([1., 1., 0.]), np.array([1., 0., 0.]), PI / 4, np.array([0, 0, 0]))

    # min_dist_dict = {
    #     ('Cd', 'Te') : 2.87046, 
    #     ('Cd', 'Cd') : 4.68745, 
    #     ('Te', 'Te') : 4.68745
    # }
    # coll_rmvl.remove_collision(struct, 0.02, min_dist_dict, fast=True,
    #     random_delete=False)
    # struct.to_vasp('col_rem_test_fst')
    # struct.to_xyz('col_rem_test_fst')
    
if __name__ == '__main__':
    main(sys.argv)