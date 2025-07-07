import math
from multiprocessing import Process, Queue
from queue import Empty
import cv2
import numpy as np
from numba import njit
from decord import VideoReader, cpu
from camera import Camera
import motion_filter as mf
from voxel_tracer import VoxelTracer
from ray import Ray
from graph import Graph

GRID_SIZE = 32
VOXEL_SIZE = 4.0

def process_camera(cam: Camera, vt: VoxelTracer, queue: Queue) -> None:
    vr = VideoReader(cam.video, cpu(0))

    # Get camera direction vector
    cam_rot = rotationMatrix(*cam.rotation)

    frame_idx = 0
    prev = cv2.cvtColor(vr[frame_idx].asnumpy(), cv2.COLOR_BGR2GRAY)

    for i in range(frame_idx + 1, len(vr)):
        frame = vr[i].asnumpy()
        next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        motion_mask = mf.filter_motion(prev, next, 2)
        
        ind = cv2.findNonZero(motion_mask)
        if ind is None: continue    # Skip frames without motion
        ind = ind.squeeze()
        ind = ind.reshape((-1, 2))  # Handle cases with only 1 coordinate pair

        raycast_intersections = []
        data = []

        pixel_centers = (cam.pixel00_loc 
                        + (ind[..., 0:1] * cam.pixel_delta_u) 
                        + (ind[..., 1:2] * cam.pixel_delta_v))
        pixel_dirs = (pixel_centers - cam.position) @ cam_rot.T
        
        for j, pixel_dir in enumerate(pixel_dirs):
            r = Ray(cam.position, pixel_dir)
            voxels = vt.raycast_into_voxels(r)
            if voxels:
                raycast_intersections.append(voxels)
                d = np.full(len(voxels), motion_mask[ind[j][1], ind[j][0]])
                data.append(d)

        queue.put((frame_idx, 
                   np.concatenate(raycast_intersections),
                   np.hstack(data)))
        frame_idx += 1
        prev = next
    # End processing
    queue.put((None, None, None))
    
def process_collector(num_processes: int, input: Queue, output: Queue, vt: VoxelTracer):
    frame_data = {}
    ended_processes = 0
    current_frame = 0
    
    while True:
        try:
            frame_idx, *values = input.get_nowait()
        except Empty:
            continue
        
        if frame_idx is None:
            ended_processes += 1
            if ended_processes == num_processes:
                break   # All processes done
            continue
        
        if frame_idx not in frame_data:
            frame_data[frame_idx] = []
        frame_data[frame_idx].append(values)
        
        try:
            if len(frame_data[current_frame]) == num_processes:
                for value in frame_data[current_frame]:
                    vt.add_motion_data(*value)
                output.put(vt.voxel_grid)
                vt.clear_motion_data()
                del frame_data[current_frame]
                current_frame += 1
        except KeyError:
            continue

def main():
    cam_L = Camera((39.694, -211.93, 1.111), 
                   (94.8, 0.000014, 13.6), 
                   "./videos/cam_L.mkv",
                   39.6)
    cam_R = Camera((72.616, 62.409, 0.047733), 
                   (90.267, -0.000012, 128.27), 
                   "./videos/cam_R.mkv",
                   39.6)
    cam_F = Camera((-133.461, 78.7308, 57.4486),
                   (69.5268, 0.000026, -120.23),
                   './videos/cam_F.mkv',
                   39.6)
    cams = [cam_L, cam_R, cam_F]
    input = Queue()
    output = Queue()
    vt = VoxelTracer(GRID_SIZE, VOXEL_SIZE)
    graph = Graph()

    processes = [Process(target=process_camera, args=(cam, vt, input)) 
                for cam in cams]
    
    collector = Process(target=process_collector,
                        args=(len(cams), input, output, vt))
    
    for cam in cams:
        cam_rot = rotationMatrix(*cam.rotation)
        cam_dir = cam_rot @ np.array((0, 0, 1))
        # Add green line representing camera direction in world space
        graph.add_ray(Ray(cam.position, cam_dir), '#00FF00', reversed=True)
        # Add yellow line representing the direction to the origin from the camera
        graph.add_ray(Ray(cam.position, cam.position - np.array((0, 0, 0))), '#FFFF00', reversed=True)
    graph.show()

    for p in processes:
        p.start()
    collector.start()
    
    while any(p.is_alive() for p in processes):
        try:
            graph.add_voxels(output.get_nowait(), vt.voxel_origin, VOXEL_SIZE)
            graph.update()
        except Empty:
            continue

    for p in processes:
        p.join()

    # motion_voxels = graph.extract_percentile_index(vt.voxel_grid, 99.9)

@njit
def rotationMatrix(x: float, y: float, z: float) -> np.ndarray:
    """Converts from Euler Angles (XYZ order) to a rotation matrix"""
    # Calculate trig values once
    cx = math.cos(x)
    sx = math.sin(x)
    cy = math.cos(y)
    sy = math.sin(y)
    cz = math.cos(z)
    sz = math.sin(z)

    # Form individual rotation matrices
    rx = np.array([[1., 0., 0.],
                    [0., cx, -sx],
                    [0., sx, cx]])
    ry = np.array([[cy, 0., sy],
                    [0., 1., 0.],
                    [-sy, 0., cy]])
    rz = np.array([[cz, -sz, 0.],
                    [sz, cz, 0.],
                    [0., 0., 1.]])

    # Form the final rotation matrix
    r = rz @ ry @ rx

    return r

if __name__ == "__main__":
    main()