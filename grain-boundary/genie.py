import sys
import os
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
    struct_1.grow_to_supercell(lattice, max_atom / 2)
    struct_2.grow_to_supercell(lattice, max_atom / 2)
    struct_1.to_vasp('grown_1')
    struct_2.to_vasp('grown_2')
    glued_struct = Structure.combine_structures(struct_1, struct_2)
    glued_struct.to_vasp('grow_and_glue')
    glued_struct.to_xyz('grow_and_glue')

    min_dist_dict = {
        ('Cd', 'Te') : 2.5, 
        ('Cd', 'Cd') : 4.0, 
        ('Te', 'Te') : 4.0
    }
    coll_rmvl.remove_collision(glued_struct, 0.025, min_dist_dict, 
        fast=False, random_delete=False)
    glued_struct.to_xyz('col_rem_test')
    glued_struct.to_vasp('col_rem_test')

    return 0

def genie(conf):
    # First read in the input files.
    if conf.struct_1 == conf.struct_2:
        orig_1 = Structure.from_file(conf.struct_1, 
                                     view_agl_count=conf.view_agl_count)
        orig_2 = copy.deepcopy(orig_1)
    else:
        orig_1 = Structure.from_file(conf.struct_1, 
                                     view_agl_count=conf.view_agl_count)
        orig_2 = Structure.from_file(conf.struct_2,
                                     view_agl_count=conf.view_agl_count)

    # Calculate min and max volume based on 
    min_vol = 0.5 * conf.atom_count_range[0] * \
        np.linalg.det(struct.coordinates)/ len(struct.direct)
    max_vol = 0.5 * conf.atom_count_range[1] * \
        np.linalg.det(struct.coordinates)/ len(struct.direct)

    # Check and create folder for output files.
    if len(conf.output_dir) != 0:
        if not os.path.isdir(conf.output_dir):
            os.mkdir(conf.output_dir)

    # For each configuration in gb_settings, produce the simulated structures.
    for [orien_1, orien_2, twist_agl] in conf.gb_settings:
        # Make deep copies of the original structures.
        struct_1 = copy.deepcopy(orig_1)
        struct_2 = copy.deepcopy(orig_2)

        try:
            # Generate transformation matrices.
            trans_1 = np.dot(geom.rotation_angle_matrix(
                np.array([0., 0., 1.]), twist_agl), geom.get_rotation_matrix(
                orien_1, np.array([0., 0., 1.])))
            trans_2 = geom.get_rotation_matrix(orien_2, np.array([0., 0., 1.]))
            # Transform structures.
            struct_1.transform(trans_1)
            struct_2.transform(trans_2)
            # Find coincident points.
            coincident_pts = coin_srch.find_coincident_points(
                struct_1.coordinates, struct_2.coordinates, 
                conf.coincident_pts_search_step, conf.coincident_pts_tolerance)
            lattice = coin_srch.find_overlattice(coincident_pts,
                conf.lattice_vec_agl_range[0], conf.lattice_vec_agl_range[1], 
                min_vol, max_vol, max_pts=max_coincident_pts_searched, 
                conf.min_vec_len=conf.min_vec_len)

            count = 0
            # Generate for each qualified lattice vector set.
            for box in lattice: 
                count += 1
                s_1_cpy = copy.deepcopy(struct_1)
                s_2_cpy = copy.deepcopy(struct_2)

                try:
                    # Grow to super-cell.
                    s_1_cpy.grow_to_supercell(box, 
                        conf.atom_count_range[1] / 2)
                    s_2_cpy.grow_to_supercell(box, 
                        conf.atom_count_range[1] / 2)
                    # Combine two structures.
                    combined_struct = Structure.combine_structures(struct_1, 
                        struct_2)

                    if not conf.skip_collision_removal:
                        # Collision removal routine.
                        coll_rmvl.remove_collision(combined_struct, 
                            conf.boundary_radius, conf.min_atom_dist, 
                            fast=conf.fast_removal, 
                            random_delete=conf.random_delete_atom)

                    # Update names and output.
                    file_name, struct_name = generate_name(conf, orien_1, orien_2, twist_agl, count)
                    combined_struct.comment = struct_name
                    combined_struct.to_file(file_name, conf.output_format, 
                        **conf.options, 
                        overwrite_protect=conf.overwrite_protect)

                except Exception, e:
                    print(str(e))
                else:
                    pass
        except Exception, e:
            print(str(e))
        else:
            pass

def generate_name(conf, orien_1, orien_2, twist_agl, count):
    struct_1_name = conf.struct_1.split('.')
    if len(struct_1) > 1:
        struct_1.name = '_'.join(struct_1_name[0:-1])
    else:
        struct_1.name = conf.struct_1

    struct_2_name = conf.struct_2.split('.')
    if len(struct_2) > 1:
        struct_2.name = '_'.join(struct_2_name[0:-1])
    else:
        struct_2.name = conf.struct_2

    trans_name = [''.join(str(orien_1.tolist()).split()), 
                  ''.join(str(orien_2.tolist()).split()),
                  str(np.rad2deg(twist_agl)), str(count)]
    trans_name = '_'.join(trans_name)

    struct_name = '_'.join[struct_1_name, struct_2_name, trans_name]
    file_name = os.path.join(conf.output_dir, struct_name)
    return file_name, struct_name

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