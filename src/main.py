import math
import cv2
import numpy as np
from camera import Camera
import motion_filter as mf
from voxel_tracer import VoxelTracer
from ray import Ray
from graph import Graph

GRID_SIZE = 32
VOXEL_SIZE = 4.0

def main():
    cam_L = Camera((39.694, -211.93, 1.111), 
                   (94.8, 0.000014, 13.6), 
                   "./videos/cam_L.mkv",
                   39.6)
    cam_R = Camera((72.616, 62.409, 0.047733), 
                   (90.267, -0.000012, 128.27), 
                   "./videos/cam_R.mkv",
                   39.6)
    cameras = [cam_L, cam_R]
    
    vt = VoxelTracer(GRID_SIZE, VOXEL_SIZE)
    graph = Graph()

    for cam in cameras:
        cap = cv2.VideoCapture(cam.video)

        ret, frame = cap.read()

        height, width, _ = frame.shape

        # Camera constants
        focal_length = 1.0
        # Viewport height constant is an arbitrary value
        h = math.tan(math.radians(cam.fov) / 2)
        viewport_height = 1.0 * h * focal_length
        viewport_width = viewport_height * width / height

        viewport_u = np.array((viewport_width, 0, 0))
        viewport_v = np.array((0, -viewport_height, 0))
        
        pixel_delta_u = viewport_u / width
        pixel_delta_v = viewport_v / height
        
        viewport_upper_left = cam.position - np.array((0, 0, focal_length)) - viewport_u / 2 - viewport_v / 2
        pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v)
        
        prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        while (ret):
            ret, frame = cap.read()
            if not ret: break
            
            next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Get camera direction vector
            cam_rot = rotationMatrix(*cam.rotation)
            cam_dir = cam_rot @ np.array((0, 0, 1))
            # Add green line representing camera direction in world space
            graph.add_ray(Ray(cam.position, cam_dir), '#00FF00', reversed=True)
            
            motion_mask = mf.filter_motion(prev, next, 2)
            for j in range(height):
                for i in range(width):
                    # Skip pixels with no motion data
                    if motion_mask[j][i] == 0:
                        continue
                    # Cast a ray through the center of the pixel
                    pixel_center = pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v)
                    pixel_dir = pixel_center - cam.position
                    pixel_dir = cam_rot @ pixel_dir
                    r = Ray(cam.position, pixel_dir)
                    voxels = vt.raycast_into_voxels(r)
                    color = '#0000FF'   # Miss
                    if voxels:
                        vt.add_motion_data(voxels, motion_mask[j][i])
                        color = '#FF0000'   # Hit
                    graph.add_ray(r, color)
            prev = next
            graph.add_voxels(vt.voxel_grid, vt.voxel_origin, VOXEL_SIZE)
            # cv2.imshow(cam.video, motion_mask)
            break
        cap.release()
    graph.show()

def rotationMatrix(x, y, z) -> np.ndarray:
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

    return r

if __name__ == "__main__":
    main()