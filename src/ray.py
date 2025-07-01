import numpy as np

class Ray:
    # Vector describing position of ray
    origin: np.ndarray
    # Vector describing direction of ray
    dir: np.ndarray
    # Inverse of the direction vector
    inv_dir: np.ndarray
    # Normalized direction vectors
    norm_dir: np.ndarray
    norm_inv_dir: np.ndarray

    def __init__(self, ro: np.ndarray, rd: np.ndarray):
        self.origin = ro
        self.dir = rd
        self.inv_dir = 1 / rd
        self.norm_dir = normalize(rd)
        self.norm_inv_dir = normalize(self.inv_dir)
    
def normalize(vector: np.ndarray) -> np.ndarray:
        """Returns the normalized vector"""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm