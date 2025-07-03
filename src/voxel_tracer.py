import numpy as np
from ray import Ray

class VoxelTracer:
    voxel_grid: np.ndarray
    voxel_size: float
    voxel_origin: np.ndarray
    grid_min: np.ndarray
    grid_max: np.ndarray
    grid_size: int

    def __init__(self, cells: int, voxel_size: float, center=np.zeros(3)):
        self.voxel_grid = np.zeros((cells, cells, cells), dtype=np.float32)
        self.voxel_size = voxel_size
        bottom_left_corner = -voxel_size * cells / 2
        self.voxel_origin = np.full(3, bottom_left_corner) + center
        self.grid_min = np.full(3, bottom_left_corner + center)
        self.grid_max = np.full(3, -bottom_left_corner + center)
        self.grid_size = cells
        
    def add_motion_data(self, voxels: list[np.ndarray], data: float):
        for v in voxels:
            self.voxel_grid[v[0]][v[1]][v[2]] += data
        
    def raycast_into_voxels(self, ray: Ray) -> list[np.ndarray]:
        """Returns all voxel indexes intersected by the raycast"""
        # Check if ray intersects voxel grid
        intersected, t_entry = self.ray_aabb(ray, self.grid_min, self.grid_max)
        if not intersected: return []

        # Initialization
        # Floating point representation of grid entry position
        start = ray.origin + ray.norm_dir * max(t_entry, 0.0)

        # Traversal constants
        step = np.sign(ray.norm_dir)
        delta = self.voxel_size / np.abs(ray.norm_dir)
        
        # Indices of current voxel
        current_voxel = np.floor((start - self.grid_min) / self.voxel_size).astype(np.int32)
        # Clamp current voxel to grid
        current_voxel = np.clip(current_voxel, 0, self.grid_size - 1)

        # Get next voxel boundary
        next_voxel = self.grid_min + (current_voxel + (step > 0)) * self.voxel_size

        # Calculate tMax, distance to the next voxel boundary for each axis
        tMax = (next_voxel - ray.origin) / ray.norm_dir
        # Handle division by zero
        tMax[ray.norm_dir == 0] = np.inf

        # Traversal
        voxels = [current_voxel.copy()]

        while (True):
            # Find which axis has the smallest tMax and traverse on that axis
            if (tMax[0] < tMax[1] and tMax[0] < tMax[2]):
                current_voxel[0] += step[0]
                if (current_voxel[0] < 0 or current_voxel[0] >= self.grid_size): break
                tMax[0] += delta[0]
            elif (tMax[1] < tMax[2]):
                current_voxel[1] += step[1]
                if (current_voxel[1] < 0 or current_voxel[1] >= self.grid_size): break
                tMax[1] += delta[1]
            else:
                current_voxel[2] += step[2]
                if (current_voxel[2] < 0 or current_voxel[2] >= self.grid_size): break
                tMax[2] += delta[2]
            voxels.append(current_voxel.copy())
        return voxels

    
    def ray_aabb(self, ray: Ray, boxMin: np.ndarray, boxMax: np.ndarray) -> tuple[bool, float]:
        """Returns whether a Ray intersects an Axis-aligned Bounding Box (AABB)
        and the time of intersection"""
        inv_dir = 1.0 / ray.norm_dir
        t1 = (boxMin[0] - ray.origin[0]) * inv_dir[0]
        t2 = (boxMax[0] - ray.origin[0]) * inv_dir[0]
        
        tmin = min(t1, t2)
        tmax = max(t1, t2)
        
        for axis in range(1, ray.origin.size):
            t1 = (boxMin[axis] - ray.origin[axis]) * inv_dir[axis]
            t2 = (boxMax[axis] - ray.origin[axis]) * inv_dir[axis]

            # Modified from original behavior to handle NaNs
            tmin = max(tmin, min(min(t1, t2), tmax))
            tmax = min(tmax, max(max(t1, t2), tmin))

        return tmax > max(tmin, 0.0), tmin

    