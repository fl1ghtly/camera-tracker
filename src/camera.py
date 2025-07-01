import math
import numpy as np

class Camera:
	"""Mimic real world camera data for testing purposes"""
	position: np.ndarray
	# Rotation in radians
	rotation: np.ndarray
	rotation_vec: np.ndarray
	video: str
    
	def __init__(self, position: tuple[float, float, float], rotation: tuple[float, float, float], video: str):
		self.position = np.array(position)
		deg2rad = np.vectorize(math.radians)
		self.rotation = deg2rad(rotation)
		self.rotation_vec = euler_to_vector(self.rotation[0], self.rotation[1], self.rotation[2])
		self.video = video
  
def euler_to_vector(x, y, z) -> np.ndarray:
    """Converts from Euler Angles (XYZ order) to a vector"""
    # Calculate trig values once
    cx = math.cos(x)
    sx = math.sin(x)
    cy = math.cos(y)
    sy = math.sin(y)
    cz = math.cos(z)
    sz = math.sin(z)

    # Form individual rotation matrices
    rx = np.array([[1, 0, 0],
                    [0, cx, -sx],
                    [0, sx, cx]])
    ry = np.array([[cy, 0, sy],
                    [0, 1, 0],
                    [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0],
                    [sz, cz, 0],
                    [0, 0, 1]])

    # Form the final rotation matrix
    r = rz @ ry @ rx

    # Forward reference vector
    v = np.array((0, 0, 1))
    
    return r @ v