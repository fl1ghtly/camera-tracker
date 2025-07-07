import numpy as np
from numba import njit, float64
from numba.experimental import jitclass

spec = [
    ('origin', float64[:]),
    ('dir', float64[:]),
    ('norm_dir', float64[:])
]

@jitclass(spec) # type: ignore
class Ray:
    """Contains information about 1 ray with dimension M
    
    Attributes:
        origin: Ray Origin (1, M)
        dir: Ray Direction (1, M)
        norm_dir: Normalized Ray Direction (1, M)
    """
    # Vector describing position of ray
    origin: np.ndarray
    # Vector describing direction of ray
    dir: np.ndarray
    # Normalized direction vectors
    norm_dir: np.ndarray

    def __init__(self, ro: np.ndarray, rd: np.ndarray):
        self.origin = ro
        self.dir = rd
        self.norm_dir = normalize(rd)

@njit
def normalize(vector: np.ndarray) -> np.ndarray:
        """Returns the normalized vector(s)
        
        Args:
            vector: (N, M) vector(s) that need to be normalized
            
        Returns:
            The normalized vector(s)
        """
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm