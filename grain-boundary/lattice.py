import numpy as np


class Lattice(object):
    """A class representing a lattice plane.
    
    Attributes:
        direction (nparray): A nparray of length 3 to represent the normal 
            vector of the plane.
        distance (float): Distance from the origin.
    """
    def __init__(self, miller, distance):
        """Initializes a Lattice object.
        
        Args:
            miller (list): List of length 3 to represent miller indices.
            distance (float): Distance from the origin.
        
        Raises:
            ValueError: Raised when argument miller has length other than 3.
        """
        if (len(miller) != 3):
            raise ValueError("Miller indexes must have three components.")
        self.direction = np.array(map(float, miller))
        self.direction = self.direction
        self.distance = float(distance)

    def __str__(self):
        """The to-string method.
        
        Returns:
            str: A string showing all the attributes of an object.
        """
        res = ""
        res += "=== LATTICE: \n"
        res += ("*** Direction: \n  (%.5f, %.5f, %.5f)\n" %
                (self.direction[0], self.direction[1], self.direction[2]))
        res += "*** Distance from Origin: \n  %.5f" % self.distance
        return res

    @staticmethod
    def from_file(path):
        """Parses a file and creates a Lattice object as specified.
        
        Args:
            path (str): The path to the input file.
        
        Returns:
            Lattice: A new Lattice object.
        
        Raises:
            ValueError: Raised when the input file is of bad format.
        """
        with open(path, 'r') as in_file:
            next_line = in_file.readline().split(';')
            in_file.close()
            if (len(next_line) != 3):
                raise ValueError("Wrong input format.\nUsage: a, b, c; d")
            direction = next_line[0].split(',')
            distance = next_line[1]
        return Lattice(direction, distance)


def main():
    lat = Lattice([1, 2, -1], 0.3)
    print(lat)

if __name__ == '__main__':
    main()
