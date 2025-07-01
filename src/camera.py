import numpy as np

class Camera:
	"""Mimic real world camera data for testing purposes"""
	position: tuple[float, float, float] 		# Meters
	rotation: tuple[float, float, float]		# Degrees
	video: str
    
	def __init__(self, position: tuple, rotation: tuple, video: str):
		self.position = position
		self.rotation = rotation
		self.video = video