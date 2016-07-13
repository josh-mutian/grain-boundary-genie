import numpy as np
import utilities as utils
import sys
from structure import Structure 

def parse_vasp(path):
    input_file = open(path, 'r')
    comment = input_file.readline().split()[0]
    scaling = float(input_file.readline())
    coordinate = np.matrix([map(float, input_file.readline().split()), 
        map(float, input_file.readline().split()), 
        map(float, input_file.readline().split())])
    
    next_line = input_file.readline().split()
    try:
        element_count = map(int, next_line)
    except ValueError as e:
        element_list = next_line
        next_line = input_file.readline().split()
        element_count = map(int, next_line)
        element_provided = False
    else:
        element_list = map(str, range(len(element_count)))
        element_provided = True
    finally:
        if (len(element_list) != len(element_count)):
            raise ValueError("Element list and count lengths should match.")

    elements = [[name for _ in range(count)] for (name, count) in 
        zip(element_list, element_count)]
    elements = np.array(elements).flatten()

    next_line = input_file.readline().split()
    if (next_line[0] == "Selective"):
        next_line = input_file.readline().split()

    if (next_line[0] != "Direct" and next_line[0] != "D"):
        raise ValueError("Only Mode \"Direct\" supported.")

    atoms = []
    for _ in range(0, sum(element_count)):
        atoms.append(map(float, input_file.readline().split()[0:3]))
    atoms = np.array(zip(atoms, elements), 
        dtype = [("position", ">f4", 3), ("element", "|S5")])

    return Structure(comment, scaling, coordinate, element_provided, atoms)

def main(argv):
    struct = parse_vasp(argv[1])
    print(struct)

if __name__ == '__main__':
    main(sys.argv)