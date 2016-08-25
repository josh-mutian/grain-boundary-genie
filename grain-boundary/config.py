import numpy as np
from math import pi as PI
import utilities as util
import json


class Configuration(object):
    """A class to specify how a grain-boundary genie runs.
    
    Attributes:
        atom_count_range (tuple): The range of acceptable atom number in the 
            final structure.
        boundary_radius (float): The proportion of lattice vector length such 
            that atoms within this distance will be considered boundary atoms.
        coincident_pts_search_step (int): Number of multiples tried to 
            replicate one structure when searching for coincidence points.
        coincident_pts_tolerance (float): The tolerance of distance between 
            two points that are considered coincidence points, in angstrom.
        fast_removal (bool): When set to True, only consider boundary atoms in 
            collision removal; otherwise use minimum image convention 
            algorithm to search for each pair of atoms within the structure.
        gb_settings (list of list): A list of lists of format 
            [struct 1 orientation, struct 2 orientation, twisting angle] 
            to specify each run of the algorithm.
        lattice_vec_agl_range (tuple): The minimum and maximum angles allowed 
            between any two lattice vectors, in rad.
        max_coincident_pts_searched (int): Maximum number of coincidence 
            points considered when searching for lattice vector sets.
        min_atom_dist (dict): A dictionary where the key is tuple of atom type
            names and value is the minimum distance in angstrom.
        min_vec_length (float): Minimum length of lattice vectors.
        mutual_view_agl_tolerance (float): Tolerance of the angle between two 
            angles that are considered mutual angles, in rad.
        output_dir (str): Name of output directory.
        output_format (str): Output file extension, currently only 'vasp', 
            'xyz', and 'ems' supported.
        output_name_prefix (str): The prefix added to each output file.
        output_options (dict): Keyword arguments required for certain file 
            types.
        overwrite_protect (bool): When set to True, if the file to be written 
            exists, find a new filename instead of overwriting the original.
        random_delete_atom (bool): When set to True, shuffle atom list before 
            collision removal.
        skip_collision_removal (bool): When set to True, skip collision 
            removal routine.
        struct_1 (str): Path to the input file of a structure.
        struct_2 (str): Path to the input file of another structure.
        view_agl_count (int): Number of viewing angles to recommend for each 
            structure.
    """
    def __init__(self, struct_1, struct_2):
        """Initializer for a Configuration object.
        
        Args:
            struct_1 (str): Path to the input file of a structure.
            struct_2 (str): Path to the input file of another structure.
        """
        # Basic GB settings.
        self.struct_1 = struct_1 
        self.struct_2 = struct_2 
        self.gb_settings = []    
        self.view_agl_count = 10 
        self.mutual_view_agl_tolerance = 0.0873 #

        # Coincident point search.
        self.coincident_pts_tolerance = 1.0
        self.coincident_pts_search_step = 25

        # Lattice vector generation.
        self.max_coincident_pts_searched = 100
        self.lattice_vec_agl_range = (0, PI)
        self.min_vec_length = 0.0
        self.atom_count_range = (1000, 10000)

        # Collision removal.
        self.skip_collision_removal = False
        self.fast_removal = True
        self.min_atom_dist = {}
        self.boundary_radius = 0.01
        self.random_delete_atom = False

        # Output format.
        self.output_format = 'vasp'
        self.output_options = {}
        self.output_dir = ''
        self.output_name_prefix = ''
        self.overwrite_protect = True

    def __str__(self):
        """Generates a string representation of a Configuration object.
        
        Returns:
            str: The string representation.
        """
        return str(self.__dict__)

    @staticmethod
    def from_json_file(path):
        """Reads an input file and converts it into a Configuration object.
        
        Args:
            path (str): The path to an input file formatted as JSON.
        
        Returns:
            Configuration obj: A Configuration object with fields set according
                to the input JSON file.
        
        Raises:
            ValueError: Raised when path of structure 1 or 2 is not provided.
                (This is the very minimum information required to generate
                a Configuration object.)
        """
        # First take in the json file and parse JSON.
        with util.open_read_file(path, 'json') as input_file:
            input_json = input_file.read()
            input_file.close()
        parsed_json = json.loads(input_json)
        keys = parsed_json.keys()

        # Parse the two structures.
        if not 'struct_1' in keys:
            raise ValueError('Keyword \'struct_1\' must present'
                             ' in the JSON input file.')
        struct_1 = parsed_json['struct_1']
        if not 'struct_2' in keys:
            raise ValueError('Keyword \'struct_2\' must present'
                             ' in the JSON input file.')
        struct_2 = parsed_json['struct_2']

        config_object = Configuration(struct_1, struct_2)

        # Parse GbSettings.
        if not 'gb_settings' in keys:
            return config_object
        config_object.gb_settings = map(
            lambda x : [np.array(x[0]).astype(float), 
                        np.array(x[1]).astype(float),
                        np.deg2rad(float(x[2]))], 
            parsed_json['gb_settings'])

        # Optional value viewing angle number.
        if 'view_agl_count' in keys:
            config_object.view_agl_count = float(parsed_json['view_agl_count'])

        if 'mutual_view_agl_tolerance' in keys:
            config_object.mutual_view_agl_tolerance = \
                np.deg2rad(float(parsed_json['mutual_view_agl_tolerance']))

        # Coincident point search parameters.
        if 'coincident_pts_tolerance' in keys:
            config_object.coincident_pts_tolerance = \
                float(parsed_json['coincident_pts_tolerance'])
        if 'coincident_pts_search_step' in keys:
            config_object.coincident_pts_search_step = \
                int(parsed_json['coincident_pts_search_step'])

        # Lattice vector generation parameters.
        if 'max_coincident_pts_searched' in keys:
            config_object.max_coincident_pts_searched = \
                int(parsed_json['max_coincident_pts_searched'])
        if 'lattice_vec_agl_range' in keys:
            config_object.lattice_vec_agl_range = \
                (np.deg2rad(float(parsed_json['lattice_vec_agl_range'][0])),
                 np.deg2rad(float(parsed_json['lattice_vec_agl_range'][1])))
        if 'min_vec_length' in keys:
            config_object.min_vec_length = float(parsed_json['min_vec_length'])
        if 'atom_count_range' in keys:
            config_object.atom_count_range = \
                (float(parsed_json['atom_count_range'][0]),
                 float(parsed_json['atom_count_range'][1]))

        # Collision removal parameters.
        if 'skip_collision_removal' in keys:
            config_object.skip_collision_removal = \
                parsed_json['skip_collision_removal']
        if 'fast_removal' in keys:
            config_object.fast_removal = parsed_json['fast_removal']
        if 'min_atom_dist' in keys:
            for [atm_1, atm_2, dist] in parsed_json['min_atom_dist']:
                config_object.min_atom_dist[(atm_1, atm_2)] = float(dist)
            if len(config_object.min_atom_dist.keys()) <= 0:
                config_object.skip_collision_removal = True

        if 'boundary_radius' in keys:
            config_object.boundary_radius = \
                float(parsed_json['boundary_radius'])
        if 'random_delete_atom' in keys:
            config_object.random_delete_atom = parsed_json['random_delete_atom']

        # Output format parameters.
        if 'output_format' in keys:
            config_object.output_format = parsed_json['output_format']
        if 'output_options' in keys:
            config_object.output_options = parsed_json['output_options']
        if 'output_dir' in keys:
            config_object.output_dir = parsed_json['output_dir']
        if 'output_name_prefix' in keys:
            config_object.output_name_prefix = parsed_json['output_name_prefix']
        if 'overwrite_protect' in keys:
            config_object.overwrite_protect = parsed_json['overwrite_protect']

        return config_object