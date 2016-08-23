"""Structure provides a paradigm to represent a crystal structure.
"""
import sys
import copy
import numpy as np
import utilities as util
import geometry as geom
from constants import PERIODIC_TABLE


class Structure(object):
    """A class representing a crystal structure.
    
    Attributes:
        cartesian (bool): A record nparray, representing the atoms in the 
            Cartesian coordinates.
        comment (str): Description of the crystal structure.
        coordinates (nparray): A 3*3 nparray with each row representing a 
            lattice vector.
        direct (TYPE): A record nparray, representing the atoms in direct
            mode.
        elements (str set): A set of all element types present in the 
            structure.    
    """

    def __init__(self, comment, scaling, coordinates, atoms):
        """Initializes a new Structure object.

        Args:
            comment (str): Description of the crystal structure.
            scaling (float): The scaling.
            coordinates (nparray): A 3*3 nparray with each row representing a 
                lattice vector.
            atoms (nparray): A record type nparray with two fields: 'position' 
                representing the positions with nparray of length 3, and 
                'element' representing the name of the elements as string.
        """
        self.comment = '_'.join(comment.split())
        self.coordinates = coordinates * scaling
        epsilon = 5.0 * (10 ** -4)
        if (abs(np.linalg.det(self.coordinates) - 0.0) <= epsilon):
            raise ValueError('Coordinate lattice not valid: singular matrix.')
        self.direct = atoms
        self.cartesian = copy.deepcopy(self.direct)
        self.cartesian['position'] = np.dot(self.cartesian['position'],
                                            np.transpose(self.coordinates))
        self.elements = set(np.unique(atoms['element']))

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

    @staticmethod
    def from_file(path):
        path_split = path.split('.')
        if len(path_split) <= 0:
            typ = ''
        else:
            typ = path_split[-1]

        if typ == 'vasp':
            return Structure.from_vasp(path)
        else:
            raise ValueError('Parser for file type %s not found.' % typ)

    def to_file(self, typ, **kwargs):
        if typ == 'vasp':
            return self.to_vasp(path)
        elif typ == 'xyz':
            return self.to_xyz(path)
        elif typ == 'ems':
            return self.to_ems(path, kwargs)
        else:
            raise ValueError('Exporter for file type %s not found.' % typ)

    @staticmethod
    def from_vasp(path):
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

            if (next_line[0] != 'Direct' and next_line[0] != 'D'):
                raise ValueError('Only Mode \'Direct\' supported.')

            atoms = []
            for _ in range(0, sum(element_count)):
                atoms.append(map(float, in_file.readline().split()[0:3]))
            atoms = np.array(zip(atoms, elements),
                             dtype=[('position', '>f4', 3), ('element', '|S5')])

        return Structure(comment, scaling, coordinates, atoms)


    def to_vasp(self, path):
        out_name = path if path.split('.')[-1] == 'vasp' else path + '.vasp'
        with util.open_write_file(out_name) as out_file:
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

    def to_xyz(self, path):
        out_name = path if path.split('.')[-1] == 'xyz' else path + '.xyz'
        with util.open_write_file(out_name) as out_file:
            out_file.write(str(self.cartesian.shape[0]) + '\n')
            out_file.write(self.comment + '\n')
            rows = []
            for ent in self.cartesian:
                rows.append(['%s' % ent[1], '%.16f' % ent[0][0],
                             '%.16f' % ent[0][1], '%.16f' % ent[0][2]])
            out_file.write(util.tabulate(rows))
        return

    def to_ems(self, path, **kwargs):
        """Outputs a Structure object as .ems file.

        Args:
            path (str): Path of the output file.
            occ (float): A constant provided by the user.
            wobble (float): A constant provided by the user.

        Returns:
            (void): Does not return.
        """
        keywords = ['occ', 'wobble']
        if not all(map(lambda x : x in kwargs.keys(), keywords)):
            raise ValueError(("A keyword argument containing all of %s "
                "must be passed to the exporter to_ems().") % 
                ', '.join(keywords))
        occ = kwargs['occ']
        wobble = kwargs['wobble']

        unit_lengths = np.apply_along_axis(lambda x: np.amax(x) - np.amin(x),
                                           0, self.cartesian['position'])
        rows = []
        rows.append(['', '', '%.4f' % unit_lengths[0], '%.4f' % unit_lengths[1],
                     '%.4f' % unit_lengths[2]])
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
        with open(out_name, 'w') as out_file:
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
                np.transpose(np.linalg.inv(self.coordinates)))
        elif (according_to == 'D'):
            self.direct.sort(order='element')
            self.cartesian = copy.deepcopy(self.direct)
            self.cartesian['position'] = np.dot(self.cartesian['position'],
                                            np.transpose(self.coordinates))
        else:
            raise ValueError('Argument according_to should either be' +
                             '"C" or "D".')
        return

    def transform(self, trans_mat):
        self.coordinates = np.dot(self.coordinates, np.transpose(trans_mat))
        self.reconcile(according_to='D')
        return

    def grow_to_supercell(self, lattice_vecs, max_atoms):
        """Grow the current struct to a super cell to fill the new lattice box.
        
        Args:
            lattice_vecs (TYPE): nparray of 3*3 representing 3 new lattice 
                vectors.
            max_atoms (TYPE): Maximum number of atoms.
        
        Returns:
            (void): Does not return.
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
            shift_vector = np.dot(self.coordinates, current_pos)
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
                    enlarged_struct = np.concatenate((enlarged_struct, shifted))
                else:
                    enlarged_struct = shifted
                next_pos = map(lambda x : x + current_pos, search_dirs)
                next_pos = [p for p in next_pos if not tuple(p.tolist()) in searched_pos]
                supercell_pos += next_pos
        # Replace the coordinate system and atom positions.
        self.direct = np.unique(enlarged_struct)
        self.direct.sort(order='element')
        self.coordinates = lattice_vecs
        # Make the Cartesian coordinates consistent.
        self.reconcile(according_to='D')
        return


def main(argv):
    struct = Structure.from_vasp(argv[1])
    new_coord = np.identity(3) * 30.0
    struct.grow_to_supercell(new_coord, 10000)
    struct.to_xyz('grown')

if __name__ == '__main__':
    main(sys.argv)
