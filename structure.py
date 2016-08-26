"""Structure provides a paradigm to represent a crystal structure.
"""
import sys
import copy
import numpy as np
import utilities as util
import geometry as geom
from constants import PERIODIC_TABLE
from math import pi as PI


class Structure(object):
    """A class representing a crystal structure.

    Attributes:
        cartesian (nparray): An array of no record nparray, representing the 
            atoms in the Cartesian coordinates.
        comment (str): Description of the crystal structure.
        coordinates (nparray): A 3*3 nparray with each row representing a 
            lattice vector.
        direct (nparray): An array of no record nparray, representing the 
            atoms in the direct coordinates.
        elements (str set): A set of all element types present in the 
            structure.    
        view_agls (nparray): An array of viewing angles (n * 3).
    """

    def __init__(self, comment, scaling, coordinates, atoms, 
                 view_agl_count=10):
        """Initializes a new Structure object.

        Args:
            comment (str): Description of the crystal structure.
            scaling (float): The scaling.
            coordinates (nparray): A 3*3 nparray with each row representing a 
                lattice vector.
            atoms (nparray): A record type nparray with two fields: 'position' 
                representing the positions with nparray of length 3, and 
                'element' representing the name of the elements as string.
            view_agl_count (int, optional): Number of viewing angles searched
                and recommended.

        Raises:
            ValueError: Raised when the input coordinate system is singular.
        """
        self.comment = '_'.join(comment.split())
        self.coordinates = coordinates * scaling
        epsilon = 5.0 * (10 ** -4)
        if (abs(np.linalg.det(self.coordinates) - 0.0) <= epsilon):
            raise ValueError('Coordinate lattice not valid: singular matrix.')
        self.direct = atoms
        self.cartesian = copy.deepcopy(self.direct)
        self.cartesian['position'] = np.dot(self.cartesian['position'],
                                            self.coordinates)
        self.elements = set(np.unique(atoms['element']))
        self.view_agls = self.find_viewing_angle(view_agl_count)

    def __str__(self):
        """The to-string method.

        Returns:
            str: A string showing all the attributes of an object.
        """
        res = ''
        res += '=== STRUCTURE: \n'
        res += '*** Comment: \n  ' + self.comment + '\n'
        res += '*** Coordinates: \n'
        rows = [['', 'x', 'y', 'z']]
        rows.append(['  a'] + map(lambda x: '%.5f' % x,
                                  self.coordinates[0].tolist()))
        rows.append(['  b'] + map(lambda x: '%.5f' % x,
                                  self.coordinates[1].tolist()))
        rows.append(['  c'] + map(lambda x: '%.5f' % x,
                                  self.coordinates[2].tolist()))
        res += util.tabulate(rows) + '\n'
        res += '*** Atoms (Direct): \n'
        rows = []
        rows.append(['  a', 'b', 'c', 'element'])
        for ent in self.direct:
            rows.append(['  %.5f' % ent[0][0], '%.5f' % ent[0][1],
                         '%.5f' % ent[0][2], '%s' % ent[1]])
        res += util.tabulate(rows)
        res += '\n*** Atoms (Cartesian): \n'
        rows = []
        rows.append(['  x', 'y', 'z', 'element'])
        for ent in self.cartesian:
            rows.append(['  %.5f' % ent[0][0], '%.5f' % ent[0][1],
                         '%.5f' % ent[0][2], '%s' % ent[1]])
        res += util.tabulate(rows)
        return res

    def find_viewing_angle(self, view_agl_count):
        """Searches for and recommends viewing angles in a Structure object.

        Args:
            view_agl_count (int): Numbers of viewing angles to recommend.

        Returns:
            nparray: A list of vectors (n * 3).
        """
        dist_to_ctr = np.apply_along_axis(
            np.linalg.norm, 1, 
            self.cartesian['position'] - np.dot(
                np.array([.5, .5, .5]), 
                self.coordinates))
        ctr_atoms = self.cartesian[np.argsort(dist_to_ctr)]
        agl_count = min(len(self.cartesian) - 1, view_agl_count)
        view_agls = ctr_atoms[1:view_agl_count +
                              1]['position'] - ctr_atoms[0]['position']
        view_agls = np.apply_along_axis(geom.normalize_vector, 1, view_agls)
        return view_agls

    @staticmethod
    def find_mutual_viewing_angle(struct_1, struct_2, tol):
        # If either structure does not have view angle, return the default
        # value of [1, 0, 0].
        if len(struct_1.view_agls) <= 0 or len(struct_2.view_agls) <= 0:
            return np.array([1., 0., 0.])

        res = []
        diff = []

        for agl_1 in struct_1.view_agls:
            for agl_2 in struct_2.view_agls:
                agl_in_between = geom.angle_between_vectors(agl_1, agl_2)
                # If an angle is within the tolerance, return it.
                if agl_in_between < tol or agl_in_between > (PI - tol):
                    res.append((agl_1 + agl_2) / 2)
                    diff.append(agl_in_between)

        res = np.array(res)
        diff = np.array(diff)
        if len(res) <= 0:
            # If no such angle exists, return the first view angle of struct_1.
            return struct_1.view_agls[0]
        else:
            # Or else output the one with smallest difference in between.
            res = res[np.argsort(diff)]
            return res[0]

    @staticmethod
    def from_file(path, **kwargs):
        """A unified method to parse a file and generate Structure object.

        Args:
            path (str): Path to file.
            **kwargs (dict): Keyword arguments for potential arguments to pass
                to the functions.

        Returns:
            Structure obj: A Structure object with fields set accordingly.

        Raises:
            ValueError: Raised when extension of the path string is not
                supported.
        """
        path_split = path.split('.')
        if len(path_split) <= 0:
            typ = ''
        else:
            typ = path_split[-1]

        if typ == 'vasp':
            return Structure.from_vasp(path, **kwargs)
        else:
            raise ValueError('Parser for file type %s not found.' % typ)

    def to_file(self, path, typ, overwrite_protect=True, **kwargs):
        """A unified method to output Structure object as file.

        Args:
            path (str): Path of the output file.
            typ (str): Output type.
            overwrite_protect (bool, optional): When set to True, will 
            generate new file name if file exists instead of overwriting.
            **kwargs (dict): Keyword arguments for potential arguments to pass
                to the functions.

        Returns:
            (void): Does not return.

        Raises:
            ValueError: Raised when type of output file is not supported.
        """
        if typ == 'vasp':
            self.to_vasp(path, overwrite_protect)
        elif typ == 'xyz':
            self.to_xyz(path, overwrite_protect)
        elif typ == 'ems':
            self.to_ems(path, overwrite_protect, **kwargs)
        else:
            raise ValueError('Exporter for file type %s not found.' % typ)

    @staticmethod
    def from_vasp(path, **kwargs):
        """Takes a .vasp file and generate Structure object.

        Args:
            path (str): Path to the file.
            **kwargs (dict): Supports keyword 'view_agl_count' to set how many
            viewing angles to find and recommend.

        Returns:
            Structure obj: A Structure object with fields set accordingly.

        Raises:
            ValueError: Raised when format is not in accordance with 
                expectation.
        """
        view_agl_count = 10
        if 'view_agl_count' in kwargs.keys():
            view_agl_count = kwargs['view_agl_count']
        with util.open_read_file(path, 'vasp') as in_file:
            comment = in_file.readline().split()[0]
            scaling = float(in_file.readline())
            coordinates = np.array([map(float, in_file.readline().split()),
                                    map(float, in_file.readline().split()),
                                    map(float, in_file.readline().split())])

            next_line = in_file.readline().split()
            try:
                element_count = map(int, next_line)
            except ValueError as e:
                element_list = next_line
                next_line = in_file.readline().split()
                element_count = map(int, next_line)
            else:
                raise ValueError('Element names not provided.')
            finally:
                if (len(element_list) != len(element_count)):
                    raise ValueError(
                        'Element list and count lengths mismatch.')

            elements = []
            for (name, count) in zip(element_list, element_count):
                elements += [name for _ in range(count)]
            elements = np.array(elements)

            next_line = in_file.readline().split()
            if (next_line[0] == 'Selective'):
                next_line = in_file.readline().split()

            if (next_line[0] != 'Direct' and next_line[0] != 'D' and
                next_line[0] != 'direct' and next_line[0] != 'd'):
                raise ValueError('Only Mode "Direct" supported.')

            atoms = []
            for _ in range(0, sum(element_count)):
                atoms.append(map(float, in_file.readline().split()[0:3]))
            atoms = np.array(zip(atoms, elements),
                             dtype=[('position', '>f4', 3),
                                    ('element', '|S5')])

        return Structure(comment, scaling, coordinates, atoms,
                         view_agl_count=view_agl_count)

    def to_vasp(self, path, overwrite_protect):
        """Outputs the Structure object as .vasp file.

        Args:
            path (str): Path to the file.
            overwrite_protect (bool): When set to true, avoid overwriting 
                existing files.

        Returns:
            (void): Does not return.
        """
        self.reconcile(according_to='D')
        out_name = path if path.split('.')[-1] == 'vasp' else path + '.vasp'
        with util.open_write_file(out_name, overwrite_protect) as out_file:
            out_file.write(self.comment + '\n1.0\n')
            for vector in self.coordinates:
                out_file.write(' '.join(map(str, vector.tolist())) + '\n')

            self.direct.sort(order='element')
            element_list = []
            element_count = []
            prev_element = None
            prev_count = None
            for ele in self.direct['element']:
                if (ele != prev_element):
                    if prev_element is not None:
                        element_list.append(prev_element)
                        element_count.append(prev_count)
                    prev_element = ele
                    prev_count = 1
                else:
                    prev_count += 1
            element_list.append(prev_element)
            element_count.append(prev_count)
            out_file.write(' '.join(element_list) + '\n')
            out_file.write(' '.join(map(str, element_count)) + '\n')

            out_file.write('Direct\n')
            for pos in self.direct['position']:
                out_file.write('%.16f  %.16f  %.16f\n' %
                               (pos[0], pos[1], pos[2]))

        return

    def to_xyz(self, path, overwrite_protect):
        """Outputs the Structure object as .xyz file.

        Args:
            path (str): Path to the file.
            overwrite_protect (bool): When set to true, avoid overwriting 
                existing files.

        Returns:
            (void): Does not return.
        """
        self.reconcile(according_to='D')
        out_name = path if path.split('.')[-1] == 'xyz' else path + '.xyz'
        with util.open_write_file(out_name, overwrite_protect) as out_file:
            out_file.write(str(self.cartesian.shape[0]) + '\n')
            out_file.write(self.comment + '\n')
            rows = []
            for ent in self.cartesian:
                rows.append(['%s' % ent[1], '%.16f' % ent[0][0],
                             '%.16f' % ent[0][1], '%.16f' % ent[0][2]])
            out_file.write(util.tabulate(rows))
        return

    def to_ems(self, path, overwrite_protect, **kwargs):
        """Outputs a Structure object as .ems file.

        Args:
            path (str): Path of the output file.
            overwrite_protect (bool): When set to true, avoid overwriting 
                existing files.
            **kwargs (dict): Keyword arguments 'occ' and 'wobble' must be
                provided.

        Returns:
            (void): Does not return.

        Raises:
            ValueError: Raised when required keyword arguments are not present
        """
        keywords = ['occ', 'wobble']
        if not all(map(lambda x: x in kwargs.keys(), keywords)):
            raise ValueError(("A keyword argument containing all of %s "
                              "must be passed to the exporter to_ems().") %
                             ', '.join(keywords))
        occ = kwargs['occ']
        wobble = kwargs['wobble']

        unit_lengths = np.apply_along_axis(lambda x: np.amax(x) - np.amin(x),
                                           0, self.cartesian['position'])
        rows = []
        rows.append(['', '', '%.4f' % unit_lengths[0],
                     '%.4f' % unit_lengths[1], '%.4f' % unit_lengths[2]])
        local_dict = {}
        for ele in self.elements:
            local_dict[ele] = PERIODIC_TABLE[ele]
        for ent in self.cartesian:
            rows.append(['', str(local_dict[ent['element']]),
                         '%.4f' % (ent['position'][0] / unit_lengths[0]),
                         '%.4f' % (ent['position'][1] / unit_lengths[1]),
                         '%.4f' % (ent['position'][2] / unit_lengths[2]),
                         '%.1f' % occ, '%.3f' % wobble])
        out_name = path if path.split('.')[-1] == 'ems' else path + '.ems'
        with util.open_write_file(out_name, overwrite_protect) as out_file:
            out_file.write(self.comment + '\n')
            out_file.write(util.tabulate(rows))
            out_file.write('\n  -1')
            out_file.close()
        return

    def reconcile(self, according_to='C'):
        """Keep direct and cartesian fields of a Structure object consistent.

        Args:
            according_to (str, optional): Description

        Returns:
            TYPE: Description

        Raises:
            ValueError: Description
        """
        if (according_to == 'C'):
            self.cartesian.sort(order='element')
            self.direct = copy.deepcopy(self.cartesian)
            self.direct['position'] = np.dot(self.cartesian['position'],
                                             np.linalg.inv(self.coordinates))
        elif (according_to == 'D'):
            self.direct.sort(order='element')
            self.cartesian = copy.deepcopy(self.direct)
            self.cartesian['position'] = np.dot(self.cartesian['position'],
                                                self.coordinates)
        else:
            raise ValueError('Argument according_to should either be' +
                             '"C" or "D".')
        return

    def transform(self, trans_mat):
        """Applies transform matrix to Structure object.

        Args:
            trans_mat (nparray): Transform matrix (3 * 3), must have 
                determinant of 1.

        Returns:
            (void): Doew not return.
        """
        assert (np.linalg.det(trans_mat) - 1.0) < 1e-3
        self.coordinates = np.dot(self.coordinates, np.transpose(trans_mat))
        self.view_agls = np.dot(self.view_agls, np.transpose(trans_mat))
        self.reconcile(according_to='D')
        return

    def grow_to_supercell(self, lattice_vecs, max_atoms):
        """Grow the current struct to a super cell to fill the new lattice box.

        Args:
            lattice_vecs (nparray): nparray of 3*3 representing 3 new lattice 
                vectors.
            max_atoms (int): Maximum number of atoms.

        Returns:
            (void): Does not return.

        Raises:
            ValueError: Raised when grown structure is empty.
        """
        # First calculate the inverse while strengthening the diagonal.
        new_coord_inv = np.linalg.inv(lattice_vecs + np.identity(3) * 1e-5)
        # supercell_pos and search_dirs are from original LabView code.
        supercell_pos = map(np.array, [
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0],
            [0, -1, 0], [0, 0, -1]
        ])
        search_dirs = map(np.array, [
            [1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0],
            [0, 0, -1], [1, 1, 0], [1, -1, 0], [-1, 1, 0], [0, 1, 1],
            [0, 1, -1], [0, -1, 1], [1, 0, 1], [1, 0, -1], [-1, 0, 1],
            [1, 1, 1], [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]
        ])
        searched_pos = set()
        enlarged_struct = []
        while len(enlarged_struct) <= max_atoms and len(supercell_pos) > 0:
            current_pos = supercell_pos.pop(0)
            if (tuple(current_pos.tolist()) in searched_pos):
                # If we have searched the position, just skip.
                continue
            searched_pos.add(tuple(current_pos.tolist()))
            shift_vector = np.dot(current_pos, self.coordinates)
            shifted = copy.deepcopy(self.cartesian)
            shifted['position'] += shift_vector
            # Convert the shifted vectors into direct with respect to the
            # new coordinate system given by lattice_vec
            shifted['position'] = np.dot(shifted['position'], new_coord_inv)
            # Filter out the atoms that are contained by the new coordinate
            # system.
            shifted = shifted[np.apply_along_axis(geom.valid_direct_vec,
                                                  1, shifted['position'])]
            if len(shifted) > 0:
                if len(enlarged_struct) > 0:
                    enlarged_struct = np.concatenate(
                        (enlarged_struct, shifted))
                else:
                    enlarged_struct = shifted
                next_pos = map(lambda x: x + current_pos, search_dirs)
                next_pos = [p for p in next_pos if not tuple(
                    p.tolist()) in searched_pos]
                supercell_pos += next_pos
        # Replace the coordinate system and atom positions.
        self.direct = np.unique(enlarged_struct)
        if len(self.direct) <= 0:
            raise ValueError('Grown super-cell is empty')
        self.direct.sort(order='element')
        self.coordinates = lattice_vecs
        # Make the Cartesian coordinates consistent.
        self.reconcile(according_to='D')
        return

    @staticmethod
    def combine_structures(struct_1, struct_2):
        """Combines two Structure objects with the same coordinates: place the 
            two Structures objects next to each other along the c axis.

        Args:
            struct_1 (Structure obj): One Structure object.
            struct_2 (Structure obj): Another Structure object.

        Returns:
            Structure obj: The combined structure.
        """
        struct_1.direct['position'][:, 2] /= 2.0
        struct_2.direct['position'][:, 2] /= 2.0
        struct_2.direct['position'][:, 2] += 0.5
        struct_1.direct = np.concatenate((struct_1.direct, struct_2.direct))
        struct_1.coordinates[2] *= 2.0
        struct_1.comment = struct_1.comment + '_' + struct_2.comment
        struct_1.reconcile(according_to='D')
        return struct_1
