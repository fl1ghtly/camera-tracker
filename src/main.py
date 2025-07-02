import math
import cv2
import numpy as np
from camera import Camera
import motion_filter as mf
from voxel_tracer import VoxelTracer
from ray import Ray

def main():
    cam_L = Camera((39.694, -211.93, 1.111), 
                   (94.8, 0, 13.6), 
                   "./videos/cam_L.mkv")
    cam_R = Camera((72.616, 62.409, 0.047733), 
                   (90.267, 0, 128.27), 
                   "./videos/cam_R.mkv")
    cameras = [cam_L]
    
    # TODO set first camera data to 0 and adjust all other camera's data accordingly
    vt = VoxelTracer(32)
    '''
    r1 = Ray(cam_L.position, cam_L.rotation_vec)
    r2 = Ray(cam_R.position, cam_R.rotation_vec)
    voxels = vt.raycast_into_voxels(r1)
    voxels = vt.raycast_into_voxels(r2)
    vt._add_motion_data(voxels, 1)
    vt._add_line(r1)
    vt._add_line(r2)
    vt._visualize_grid()
    '''
    for cam in cameras:
        cap = cv2.VideoCapture(cam.video)

        ret, frame = cap.read()

        height, width, _ = frame.shape

        # Camera constants
        focal_length = 1.0
        # Viewport height is an arbitrary value
        viewport_height = 2.0
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
            cam_rot = rotationMatrix(*cam.rotation)
            cam_dir = cam_rot @ np.array((0, 0, 1))
            vt._add_line(Ray(cam.position, cam_dir), '#00FF00')
            # motion_mask = mf.filter_motion(prev, next, 2)
            for j in range(0, height, 100):
                for i in range(0, width, 100):
                    pixel_center = pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v)
                    pixel_dir = pixel_center - cam.position
                    # pixel_dir = rotationMatrix(-math.pi / 2, 0, 0) @ pixel_dir
                    pixel_dir = cam_rot @ pixel_dir
                    r = Ray(cam.position, -pixel_dir)
                    '''
                    voxels = vt.raycast_into_voxels(r)
                    if voxels:
                        vt._add_motion_data(voxels, pixel)
                    '''
                    vt._add_line(r, '#FF0000')
            prev = next
            break
        cap.release()

    vt._visualize_grid()

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