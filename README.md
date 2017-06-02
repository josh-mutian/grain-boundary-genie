### Grain Boundary Genie

The grain boundary genie, now in Python, is based on Christopher Buurma's thesis [_Application of ab-initio Methods to Grain Boundaries and Point Defects for Poly-CdTe Solar Cells_](https://www.researchgate.net/publication/278727948_Application_of_ab-initio_Methods_to_Grain_Boundaries_and_Point_Defects_for_Poly-CdTe_Solar_Cells?enrichId=rgreq-7eaf52f248418b5e3d5f3f2a853f1b2c-XXX&enrichSource=Y292ZXJQYWdlOzI3ODcyNzk0ODtBUzoyNDIxNTY0MzQ2MjA0MTZAMTQzNDc0NjAwMTc3OA%3D%3D&el=1_x_3) and his original [implementation in LabView](https://github.com/Fermi-Dirac/Vasp-Helper).

In this new implementation, the program is restructured so that the it will be easier to understand. Vectorization through Numpy library enables the code to run much faster, in face of myriad amount of crystal structures needed for further simulations and experiments. Also, the Python implementation gives user many freedom to specify how the code runs: users can feed the configurations in a JSON file, which provides nice serializations of lists and dictionaries so that more complicated data structures can be input by the users in a tidy fashion.

Documentation of the genie is still in progress and a Wiki on GitHub is expected.

If you have any inquiries, please contact lium [at] anl [dot] gov.

#### Usage
Please refer to the file `example_input.json` to construct a `.json` file of the configuration that you want to run. Then `cd` to the directory of the ginie and run `python genie.py *.json` where `*.json` is to be replaced by the `.json` configuration file that you have just constructed.