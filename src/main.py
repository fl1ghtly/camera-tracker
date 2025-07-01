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
    cameras = [cam_L, cam_R]
    
    # TODO set first camera data to 0 and adjust all other camera's data accordingly
    vt = VoxelTracer(32)
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
        prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        while (ret):
            ret, frame = cap.read()
            if not ret: break
            
            next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            motion_mask = mf.filter_motion(prev, next, 2).astype(np.float32)
            prev = next
        
        cap.release()
    '''

if __name__ == "__main__":
    main()