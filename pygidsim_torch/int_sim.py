import torch
from torch import Tensor


class Intensity:
    """
    A class to calculate the GIWAXS intensities

    Attributes
    ----------
    atoms :
        List of elements in the structure.
    atom_positions :
        Relative atom coordinates (related with the atoms attribute).
    occ : Tensor
        Atom occupancies (related with the atoms attribute).
    q_3d : Tensor
        Peak vectors in 3d reciprocal space.
    mi :  Tensor
        Miller indices.
    wavelength :
        Beam wavelength, Å.
    ai :
        Incidence angle, degrees.
    database :
        Database with form-factors.

    Methods
    -------
    get_intensities():
        Return intensities for each peak position.
    """

    def __init__(self):
        pass
