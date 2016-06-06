"""
By: Akhila Ananthram (asa225)
"""
import numpy as np

# mappings used in multiple files
MAPPING = {"A": 0, "G": 1, "C": 2, "T": 3, "-": 4}
RETURN_MAPPING = np.array(zip(*sorted(MAPPING.iteritems(),
    key=lambda x: x[1]))[0])
NUM_BASES = len(MAPPING) - 1
