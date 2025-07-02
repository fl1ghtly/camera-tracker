import math
import numpy as np

class Camera:
	"""Mimic real world camera data for testing purposes"""
	position: np.ndarray
	# Rotation in radians
	rotation: np.ndarray
	video: str
    
	def __init__(self, position: tuple[float, float, float], rotation: tuple[float, float, float], video: str):
		self.position = np.array(position)
		deg2rad = np.vectorize(math.radians)
		self.rotation = deg2rad(rotation)
		self.video = video