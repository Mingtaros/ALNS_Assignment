import copy
import random

import numpy as np
from psp import PSP


### Destroy operators ###
# You can follow the example and implement destroy_2, destroy_3, etc
def destroy_1(current: PSP, random_state):
    """Destroy operator sample (name of the function is free to change)
    Args:
        current::PSP
            a PSP object before destroying
        random_state::numpy.random.RandomState
            a random state specified by the random seed
    Returns:
        destroyed::PSP
            the PSP object after destroying
    """
    destroyed = current.copy()
    # // Implement Code Here


### Repair operators ###
# You can follow the example and implement repair_2, repair_3, etc
def repair_1(destroyed: PSP, random_state):
    """repair operator sample (name of the function is free to change)
    Args:
        destroyed::PSP
            a PSP object after destroying
        random_state::numpy.random.RandomState
            a random state specified by the random seed
    Returns:
        repaired::PSP
            the PSP object after repairing
    """
    # // Implement Code Here
