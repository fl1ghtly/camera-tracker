import math
import numpy as np
import pyvista as pv
from ray import Ray

class VoxelTracer:
    voxel_grid: np.ndarray
    voxel_size: float
    voxel_origin: np.ndarray
    grid_min: np.ndarray
    grid_max: np.ndarray
    grid_size: int
    plotter: pv.Plotter

    def __init__(self, cells: int, voxel_size: float):
        self.voxel_grid = np.zeros((cells, cells, cells), dtype=np.float32)
        self.voxel_size = voxel_size
        bottom_left_corner = -voxel_size / 2 * cells
        self.voxel_origin = np.full(3, bottom_left_corner)
        self.grid_min = np.full(3, -cells / 2)
        self.grid_max = np.full(3, cells / 2)
        self.grid_size = cells
        self.plotter = pv.Plotter()
        
    def _visualize_grid(self):

        grid = pv.ImageData()
        grid.dimensions = np.array(self.voxel_grid.shape) + 1
        grid.spacing = (self.voxel_size, self.voxel_size, self.voxel_size)
        grid.origin = self.voxel_origin
        grid.cell_data['values'] = self.voxel_grid.flatten(order="F")

        self.plotter.add_mesh(grid, show_edges=True)
        self.plotter.show_grid() # type: ignore
        self.plotter.show()
        
    def _add_line(self, ray: Ray, color: str, reversed=False):
        rev = -1 if reversed else 1
        line = pv.Line(ray.origin, ray.origin + ray.norm_dir * 300 * rev)
        self.plotter.add_mesh(line, color=color, line_width=2)
        
    def _add_motion_data(self, voxels: list[np.ndarray], data: float):
        for v in voxels:
            self.voxel_grid[v[0]][v[1]][v[2]] += data
        
    def raycast_into_voxels(self, ray: Ray) -> list[np.ndarray]:
        """Returns all voxel indexes intersected by the raycast
        
        ro: Ray origin
        rd: Ray direction
        """
        # Check if ray is casted into the voxel grid
        intersected, t_entry = self.ray_aabb(ray, self.grid_min, self.grid_max)
        if not intersected: return []

        # Initialization
        # floating point representation of grid entry position
        start = ray.origin + ray.norm_dir * max(t_entry, 0.0)
        # Be lenient on possible floating point inaccuracies
        start[start == self.grid_size] = self.grid_size - 1
        
        # traversal constants
        step = np.sign(ray.norm_dir)
        delta = 1.0 / ray.norm_dir

        # indices of current voxel
        pos = np.clip(np.floor(start), 0, self.grid_size)
        
        tMax = (pos + step - start) / ray.norm_dir
        # Handle division by zero
        division_err = np.argwhere(ray.norm_dir == 0)
        np.put(tMax, division_err, np.inf)
        
        # Traversal
        voxels = [np.array(pos.astype(np.int32))]

        while (True):
            if (tMax[0] < tMax[1] and tMax[0] < tMax[2]):
                pos[0] += step[0]
                if (pos[0] < 0 or pos[0] >= self.grid_size): break
                tMax[0] += delta[0]
            elif (tMax[1] < tMax[2]):
                pos[1] += step[1]
                if (pos[1] < 0 or pos[1] >= self.grid_size): break
                tMax[1] += delta[1]
            else:
                pos[2] += step[2]
                if (pos[2] < 0 or pos[2] >= self.grid_size): break
                tMax[2] += delta[2]
            voxels.append(np.array(pos.astype(np.int32)))
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

    