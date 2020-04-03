import constants as c
import numpy as np
import scipy as sc


def r_cyl_to_spher(r_spher,r_grid,z_grid):
    for i, r in enumerate(r_grid):
        for j, z in enumerate(z_grid):
            r_spher[i,j]=np.sqrt(r**2.+z**2)
    return r_spher


def get_value(array,mgrid,radius,height):
    value = 0.0


    return value
