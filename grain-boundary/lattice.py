import numpy as np


class Lattice(object):
    """docstring for Lattice"""

    def __init__(self, miller, distance):
        if (len(miller) != 3):
            raise ValueError("Miller indexes must have three components.")
        self.direction = np.array(map(float, miller))
        self.direction = self.direction
        self.distance = float(distance)

    def __str__(self):
        res = ""
        res += "=== LATTICE: \n"
        res += ("*** Direction: \n  (%.5f, %.5f, %.5f)\n" %
                (self.direction[0], self.direction[1], self.direction[2]))
        res += "*** Distance from Origin: \n  %.5f" % self.distance
        return res

    @staticmethod
    def from_file(path):
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
