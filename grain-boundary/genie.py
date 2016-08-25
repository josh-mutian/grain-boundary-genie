import sys
import os
import copy
import numpy as np
import traceback
from structure import Structure
from config import Configuration
import geometry as geom
import collision_removal as coll_rmvl
import coincidence_search as coin_srch
from math import pi as PI


def genie(conf):
    print(conf)
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
    atom_count_unit_vol = (len(orig_1.direct) + len(orig_2.direct)) / \
        (abs(np.linalg.det(orig_1.coordinates)) + 
         abs(np.linalg.det(orig_2.coordinates)))
    min_vol = conf.atom_count_range[0] / atom_count_unit_vol
    max_vol = conf.atom_count_range[1] / atom_count_unit_vol

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
            # Find mutual viewing angle and generate a matrix that will turn 
            # the mutual viewing angle into the direction of [1, 0, 0].
            mutual_view_agl = Structure.find_mutual_viewing_angle(
                struct_1, struct_2, tol=conf.mutual_view_agl_tolerance)
            mat_turn_mutual = geom.get_rotation_matrix(
                mutual_view_agl, np.array([1., 0., 0.]))
            # Find coincident points.
            coincident_pts = coin_srch.find_coincidence_points(
                struct_1.coordinates, struct_2.coordinates, 
                conf.coincident_pts_search_step, conf.coincident_pts_tolerance)
            lattice = coin_srch.find_overlattice(coincident_pts,
                conf.lattice_vec_agl_range[0], conf.lattice_vec_agl_range[1], 
                min_vol, max_vol, max_pts=conf.max_coincident_pts_searched, 
                min_vec_len=conf.min_vec_length)

            count = 0
            # Generate for each qualified lattice vector set.
            for box in lattice: 
                print('Current lattice vector set:')
                print(box)
                print('Expected atom count: %d' % int(np.linalg.det(box) * atom_count_unit_vol))

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
                    combined_struct = Structure.combine_structures(
                        s_1_cpy, s_2_cpy)

                    # Sanity check: whether the actual atom count matches with 
                    # expected atom count.
                    if len(combined_struct.direct) < np.linalg.det(
                        combined_struct.coordinates) * \
                        atom_count_unit_vol * 0.80:
                        print('Expected atom count not met.')
                        count -= 1
                        continue


                    if not conf.skip_collision_removal:
                        # Collision removal routine.
                        coll_rmvl.remove_collision(combined_struct, 
                            conf.boundary_radius, conf.min_atom_dist, 
                            fast=conf.fast_removal, 
                            random_delete=conf.random_delete_atom)

                    # Turn the combined structure according to mutual viewing
                    # angle.
                    combined_struct.transform(mat_turn_mutual)
                    # Update names and output.
                    file_name, struct_name = generate_name(conf, orien_1, orien_2, twist_agl, count)
                    combined_struct.comment = struct_name
                    combined_struct.to_file(
                        file_name, conf.output_format, 
                        overwrite_protect=conf.overwrite_protect, 
                        **conf.output_options)
                except Exception, e:
                    traceback.print_tb(sys.exc_info()[2])
                    print(str(e))
                else:
                    pass
        except Exception, e:
            traceback.print_tb(sys.exc_info()[2])
            print(str(e))
        else:
            pass

def generate_name(conf, orien_1, orien_2, twist_agl, count):
    struct_1_name = conf.struct_1.split('/')[-1]
    struct_1_name = struct_1_name.split('.')
    if len(struct_1_name) > 1:
        struct_1_name = '_'.join(struct_1_name[0:-1])
    else:
        struct_1_name = conf.struct_1

    struct_2_name = conf.struct_2.split('/')[-1]
    struct_2_name = struct_2_name.split('.')
    if len(struct_2_name) > 1:
        struct_2_name = '_'.join(struct_2_name[0:-1])
    else:
        struct_2_name = conf.struct_2

    trans_name = [''.join(str(orien_1.tolist()).split()), 
                  ''.join(str(orien_2.tolist()).split()),
                  str(np.rad2deg(twist_agl)), str(count)]
    trans_name = '_'.join(trans_name)

    struct_name = '_'.join([struct_1_name, struct_2_name, trans_name])
    file_name = os.path.join(conf.output_dir, struct_name)
    print(file_name)
    print(struct_name)
    return file_name, struct_name

def main(argv):
    if len(argv) < 2:
        # In this case, find all .json files in the current directory.
        for conf_file in [f for f in os.listdir('.') if f.endswith('.json')]:
            try:
                genie(Configuration.from_json_file(conf_file))
            except Exception, e:
                print(str(e))
            else:
                pass
    elif os.path.isfile(argv[1]):
        # In this case, read in the file and run genie.
        genie(Configuration.from_json_file(argv[1]))
    elif os.path.isdir(argv[1]):
        # In this case, find all .json files in the given directory.
        for conf_file in [f for f in os.listdir(agrv[1]) if \
            f.endswith('.json')]:
            try:
                genie(Configuration.from_json_file(conf_file))
            except Exception, e:
                traceback.print_tb(sys.exc_info()[2])
                print(str(e))
            else:
                pass
    else:
        print('USAGE')
        sys.exit(1)

if __name__ == '__main__':
    main(sys.argv)