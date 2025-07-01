import cv2
import numpy as np
from camera import Camera
import motion_filter as mf
from voxel_tracer import VoxelTracer

def main():
    cam_L = Camera((39.694, -211.93, 1.111), 
                   (94.8, 0.0000001, 13.6), 
                   "./videos/cam_L.mkv")
    cam_R = Camera((72.616, 62.409, 0.047733), 
                   (90.267, 0.0000001, 128.27), 
                   "./videos/cam_R.mkv")
    cameras = [cam_L, cam_R]
    
    # TODO set first camera data to 0 and adjust all other camera's data accordingly
    vt = VoxelTracer(4)
    # voxels = vt.raycast_into_voxels(np.array(cam_L.position), np.array(cam_L.rotation))
    ro = np.array((0, 0, 0))
    rd = np.array((5, 3, 2))
    rd = vt.normalize(rd)
    voxels = vt.raycast_into_voxels(ro, rd)
    for v in voxels:
        print(v)
    vt._add_motion_data(voxels, 1)
    vt._visualize_grid(ro, rd)
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