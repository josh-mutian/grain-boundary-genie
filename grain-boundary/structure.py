"""Summary
"""
import sys
import copy
import numpy as np
import utilities as util
import geometry as geom
from lattice import Lattice


class Structure(object):
    """A class representing a crystal structure.

    Attributes:
        atoms (nparray): A record type nparray with two fields: 'position' 
            representing the positions with nparray of length 3, and 'element' 
            representing the name of the elements as string.        
        cartesian (bool): Set to true if the coordinate is in Cartesian.
        comment (str): Description of the crystal structure.
        coordinate (nparray): A 3*3 nparray with each row representing a 
            lattice vector.
        elements (str set): A set of all element types present in the 
            structure.
        elements_provided (bool): Whether the element names are specified in 
            the input file.
        scaling (float): The scaling.
    """

    def __init__(self, comment, scaling, coordinate, elements_provided, atoms):
        """Initializes a new Structure object.

        Args:
            comment (str): Description of the crystal structure.
            scaling (float): The scaling.
            coordinate (nparray): A 3*3 nparray with each row representing a 
                lattice vector.
            elements_provided (bool): Whether the element names are specified 
                in the input file.
            atoms (nparray): A record type nparray with two fields: 'position' 
                representing the positions with nparray of length 3, and 
                'element' representing the name of the elements as string.
        """
        self.comment = comment
        self.scaling = scaling
        self.coordinate = coordinate
        self.elements_provided = elements_provided
        self.atoms = atoms
        self.elements = set(np.unique(atoms['element']))
        self.cartesian = False

    def __str__(self):
        """The to-string method.

        Returns:
            str: A string showing all the attributes of an object.
        """
        res = ''
        res += '=== STRUCTURE: \n'
        res += '*** Comment: \n  ' + self.comment + '\n'
        res += '*** Scaling: \n  %.5f\n' % (self.scaling)
        res += '*** Coordinates: \n'
        rows = [['', 'x', 'y', 'z']]
        rows.append(['  a'] + map(lambda x: '%.5f' % x,
                                  self.coordinate[0].tolist()))
        rows.append(['  b'] + map(lambda x: '%.5f' % x,
                                  self.coordinate[1].tolist()))
        rows.append(['  c'] + map(lambda x: '%.5f' % x,
                                  self.coordinate[2].tolist()))
        res += util.tabulate(rows) + '\n'
        res += '*** Element Names: \n  '
        res += ('P' if self.elements_provided else 'Not p') + 'rovided\n'
        res += '*** In Cartesian System: \n  '
        res += ('Yes' if self.cartesian else 'No') + '\n'
        res += '*** Atoms: \n'
        rows = []
        rows.append(['  a', 'b', 'c', 'element'])
        for ent in self.atoms:
            rows.append(['  %.5f' % ent[0][0], '%.5f' % ent[0][1],
                         '%.5f' % ent[0][2], '%s' % ent[1]])
        res += util.tabulate(rows)
        return res

    def cut_by_lattice(self, lattice):
        """Cuts a Structure object by a Lattice object.

        Args:
            lattice (Lattice): The lattice plane that cuts the Structure 
                object (by deleting atoms in it.)

        Returns:
            (void): Does not return.
        """
        distance = np.dot(lattice.direction,
                          np.transpose(self.atoms['position']))
        self.atoms = self.atoms[np.where(distance < lattice.distance)]
        return

    def to_cartesian(self):
        """Turns a Structure object into Cartesian coordinate.

        Returns:
            (void): Does not return.
        """
        if self.cartesian:
            return
        new_representation = np.dot(np.transpose(self.coordinate),
                                    np.transpose(self.atoms['position']))
        self.atoms['position'] = self.scaling * \
            np.transpose(new_representation)
        self.cartesian = True
        return

    def to_coordinate(self):
        """Turns a Structure object into the coordinate provided by itself.

        Returns:
            (void): Does not return.
        """
        if not self.cartesian:
            return
        orig_representation = np.dot(np.linalg.inv(np.transpose(
            self.coordinate)), np.transpose(self.atoms['position']))
        self.atoms['position'] = np.transpose(orig_representation)
        self.cartesian = False
        return

    @staticmethod
    def from_vasp(path):
        """Reads in .vasp file and create a Structure object as specified.

        Args:
            path (str): The path to the .vasp file.

        Returns:
            Structure: The Structure object created.

        Raises:
            ValueError: Raised when the file cannot be parsed.
        """
        with open(path, 'r') as in_file:
            comment = in_file.readline().split()[0]
            scaling = float(in_file.readline())
            coordinate = np.array([map(float, in_file.readline().split()),
                                   map(float, in_file.readline().split()),
                                   map(float, in_file.readline().split())])

            next_line = in_file.readline().split()
            try:
                element_count = map(int, next_line)
            except ValueError as e:
                element_list = next_line
                next_line = in_file.readline().split()
                element_count = map(int, next_line)
                elements_provided = True
            else:
                element_list = map(str, range(len(element_count)))
                elements_provided = False
            finally:
                if (len(element_list) != len(element_count)):
                    raise ValueError(
                        'Element list and count lengths mismatch.')

            elements = [[name for _ in range(count)] for (name, count) in
                        zip(element_list, element_count)]
            elements = np.array(elements).flatten()

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

        return Structure(comment, scaling, coordinate, elements_provided, atoms)

    def to_vasp(self, name):
        """Saves the Structure object into a .vasp file.

        Args:
            name (str): The path of the output .vasp file.

        Returns:
            (void): Does not return.
        """
        with open(name, 'w') as out_file:
            out_file.write(self.comment + '\n')
            out_file.write(str(self.scaling) + '\n')
            for vector in self.coordinate:
                out_file.write(' '.join(map(str, vector.tolist())) + '\n')

            # Generate element list, assuming well ordered.
            element_list = []
            element_count = []
            prev_element = None
            prev_count = None
            for ele in self.atoms['element']:
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
            if self.elements_provided:
                out_file.write(' '.join(element_list) + '\n')
                out_file.write(' '.join(map(str, element_count)) + '\n')
            else:
                out_file.write(' '.join(element_list) + '\n')
                out_file.write(' '.join(map(str, element_count)) + '\n')

            out_file.write('Direct\n')
            for pos in self.atoms['position']:
                out_file.write('%.16f  %.16f  %.16f\n' %
                               (pos[0], pos[1], pos[2]))
        return

    def to_xyz(self, name):
        """Saves the Structure object into a .xyz file.

        Args:
            name (str): The path of the output .xyz file.

        Returns:
            (void): Does not return.
        """
        orig_format = self.cartesian
        self.to_cartesian()
        with open(name, 'w') as out_file:
            out_file.write(str(self.atoms.shape[0]) + '\n')
            out_file.write(self.comment + '\n')
            rows = []
            for ent in self.atoms:
                rows.append(['%s' % ent[1], '%.16f' % ent[0][0],
                             '%.16f' % ent[0][1], '%.16f' % ent[0][2]])
            out_file.write(util.tabulate(rows))
        if not orig_format:
            self.to_coordinate()
        return

    def rotate(self, from_vector, to_vector):
        """Rotates the atoms in a Structure object.

        Args:
            from_vector (nparray): nparray of length 3 to represent a vector.
            to_vector (nparray): nparray of length 3 to represent a vector.

        Returns:
            (void): Does not return.
        """
        # This is forced to be done in Cartesian mode.
        self.to_cartesian()
        rotation_matrix = geom.get_rotation_matrix(from_vector, to_vector)
        self.atoms['position'] = np.transpose(
            np.dot(rotation_matrix, np.transpose(self.atoms['position'])))
        # Also apply the rotation matrix to the original coordinate vectors
        # in case that they are useful in the future.
        self.coordinate = np.transpose(
            np.dot(rotation_matrix, np.transpose(self.coordinate)))
        return

    @staticmethod
    def combine(struct_1, struct_2):
        """Combines two Structure objects.

        Args:
            struct_1 (Structure): The first Structure object.
            struct_2 (Structure): The second Structure object.

        Returns:
            Structure: The combined Structure object.

        Raises:
            ValueError: Raised when one Structure object provides element 
                names but the other one does not.
        """
        # First check that elements_provided values same.
        if struct_1.elements_provided != struct_2.elements_provided:
            raise ValueError(
                'Cannot combine two structures: ',
                'They must have same information (element names) ',
                'provided or not provided.'
            )
        # Both structs should be in Cartesian mode.
        struct_1.to_cartesian()
        struct_2.to_cartesian()
        new_comment = ('Combined: ' + struct_1.comment + ' + ' +
                       struct_2.comment)
        new_atoms = np.hstack((struct_1.atoms, struct_2.atoms))
        new_struct = Structure(new_comment, 1.0, None,
                               struct_1.elements_provided, new_atoms)
        new_struct.cartesian = True
        return new_struct

    @staticmethod
    def cut_and_combine(struct_1, lattice_1, struct_2, lattice_2, d):
        """Cuts two Structure objects by two Lattice objects and combines them.

        Args:
            struct_1 (Structure): The first Structure object.
            lattice_1 (Lattice): The lattice plane that cuts struct_1.
            struct_2 (Structure): The second Structure object.
            lattice_2 (Lattice): The lattice plane that cuts struct_2.
            d (float): The distance between the two Structure objects.

        Returns:
            Structure: A cut and combined Structure object.

        Raises:
            ValueError: Raised when the two Structures have different element 
                sets. (This restriction is imposed because of the similarity
                assessment algorithm. May be relaxed if better algorithm is 
                found.)
        """
        # If two structures are in fact the same, make a deep copy so that
        # operations on one will not affect the other.
        if struct_1 == struct_2:
            struct_2 = copy.deepcopy(struct_1)
        # Check that two structures have the same set of elements.
        if struct_1.elements != struct_2.elements:
            raise ValueError(
                'Two structures should have same set of elements.')
        elements = struct_1.elements
        # First cut two structures by lattices.
        struct_1.cut_by_lattice(lattice_1)
        struct_2.cut_by_lattice(lattice_2)
        # Force into Cartesian.
        struct_1.to_cartesian()
        struct_2.to_cartesian()
        # Then rotate the two structures to face each other.
        struct_1.rotate(lattice_1.direction, [0, 0, 1])
        struct_2.rotate(lattice_2.direction, [0, 0, -1])
        # Calculate the centroids of two cutting surfaces of the two structures
        # and recenter them.
        # TODO: Discuss the method used to calculate the 'centroids.'
        epsilon = 1.0 * 10 ** -5
        cut_face_z_1 = np.amax(struct_1.atoms['position'][:, 2])
        cut_face_atoms_1 = struct_1.atoms[
            np.where(abs(struct_1.atoms['position'][:, 2] - cut_face_z_1)
                     <= epsilon)]
        cut_face_center_1 = np.mean(cut_face_atoms_1['position'], axis=0)
        struct_1.atoms['position'] -= cut_face_center_1

        cut_face_z_2 = np.amin(struct_2.atoms['position'][:, 2])
        cut_face_atoms_2 = struct_2.atoms[
            np.where(abs(struct_2.atoms['position'][:, 2] - cut_face_z_2)
                     <= epsilon)]
        cut_face_center_2 = np.mean(cut_face_atoms_2['position'], axis=0)
        struct_2.atoms['position'] -= cut_face_center_2

        # Raise struct_2 by d.
        struct_2.atoms['position'][:, 2] += abs(d)

        combined_atoms = np.concatenate((struct_1.atoms, struct_2.atoms))
        dummy_struct = Structure('Dummy Structure', 1.0, None, True,
                                 combined_atoms)
        dummy_struct.cartesian = True
        dummy_struct.to_xyz('Glued structure.xyz')

        def get_periodic_box(atoms, elements, axis, d, slice=10):
            """Gets a pair of planes so that PBC is met.

            Args:
                atoms (nparray): A record type nparray representing atoms.
                elements (str set): Set of elements present.
                axis (int): An integer representing which axis the planes are
                    perpendicular to. 1, 2, and 3 represents x-, y-, and z-axis
                    respectively.
                d (float): Distance between cutting interfaces.
                slice (int, optional): Number of slices created.

            Returns:
                float tup: A tuple of length 2 representing the position of 
                    the pair of planes.
            """
            ax_min = np.amin(atoms['position'][:, axis])
            ax_max = np.amax(atoms['position'][:, axis])
            ax_step = (ax_max - ax_min) / slice
            ax_grid = np.arange(start=ax_min + ax_step / 2,
                                slice=ax_step, stop=ax_max)
            ax_slice = map(lambda x:
                           atoms[np.where(np.logical_and(
                               atoms['position'][:, axis] >= x - ax_step / 2,
                               atoms['position'][:, axis] <= x + ax_step / 2))],
                           ax_grid)
            ax_slice = map(lambda x:
                           [x[np.where(x['element'] == e)]['position']
                            for e in elements],
                           ax_slice)
            ax_optimal_dist = float('inf')
            ax_optimal_slices = None
            for i in np.where(ax_grid <= 0)[0]:
                for j in np.where(ax_grid >= d)[0]:
                    dist = geom.slice_distances(ax_slice[i], ax_slice[j])
                    if dist < ax_optimal_dist:
                        ax_optimal_slices = (ax_grid[i] - ax_step, ax_grid[j]
                                             + ax_step)
                        ax_optimal_dist = dist
            return ax_optimal_slices

        # Get the box, cut the combined structure and recenter.
        box = [get_periodic_box(combined_atoms, elements, i, d) for i in
               [0, 1, 2]]
        # print(box)
        for i in range(3):
            combined_atoms = combined_atoms[np.where(np.logical_and(
                combined_atoms['position'][:, i] >= box[i][0],
                combined_atoms['position'][:, i] <= box[i][1]))]
        combined_atoms['position'] -= np.apply_along_axis(np.amin, 0,
                                                          combined_atoms['position'])

        # Sort atoms by element.
        combined_atoms.sort(order='element')
        # Generate new coordinate.
        new_coordinate = np.array([
            [box[0][1] - box[0][0], 0.0, 0.0],
            [0.0, box[1][1] - box[1][0], 0.0],
            [0.0, 0.0, box[2][1] - box[2][0]]
        ])
        # Normalize atom positions.
        new_struct = Structure(struct_1.comment + '+' + struct_2.comment,
                               1.0, new_coordinate, struct_1.elements_provided, combined_atoms)
        new_struct.cartesian = True
        new_struct.to_coordinate()
        return new_struct


def main(argv):
    """A main function for testing.

    Args:
        argv (str list): A list of input arguments.

    Returns:
        (void): Does not return.
    """
    struct = Structure.from_vasp(argv[1])
    lattice_1 = Lattice([1, 1, 0], 1.5)
    lattice_2 = Lattice([1, 1, 0], 1.5)
    d = 2.5
    new_struct = Structure.cut_and_combine(
        struct, lattice_1, struct, lattice_2, d)
    new_struct.to_vasp('box_test.vasp')
    new_struct.to_xyz('box_test.xyz')

if __name__ == '__main__':
    main(sys.argv)
