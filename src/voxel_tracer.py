import math
import numpy as np
import pyvista as pv

class VoxelTracer:
    voxel_grid: np.ndarray
    grid_min: np.ndarray
    grid_max: np.ndarray
    grid_size: int

    def __init__(self, n: int):
        self.voxel_grid = np.zeros((n, n, n), dtype=np.float32)
        self.grid_min = np.array((0, 0, 0))
        self.grid_max = np.array((n, n, n))
        self.grid_size = n
        
    def _visualize_grid(self, ro: np.ndarray, rd: np.ndarray):
        pl = pv.Plotter()

        grid = pv.ImageData()
        grid.dimensions = np.array(self.voxel_grid.shape) + 1
        grid.spacing = (1, 1, 1)
        grid.cell_data['values'] = self.voxel_grid.flatten(order="F")
        
        line = pv.Line(ro, ro + rd * 16)

        pl.add_mesh(grid)
        pl.add_mesh(line, color='#FF0000', line_width=10)
        pl.show_grid() # type: ignore
        pl.show()

    def _add_motion_data(self, voxels: list[np.ndarray], data: float):
        for v in voxels:
            self.voxel_grid[v[0]][v[1]][v[2]] = data
        
    def raycast_into_voxels(self, ro: np.ndarray, rd: np.ndarray) -> list[np.ndarray]:
        """Returns all voxel indexes intersected by the raycast
        
        ro: Ray origin
        rd: Ray direction
        """
        # Check if ray is casted into the voxel grid
        intersected, t_entry = self.ray_aabb(ro, rd, self.grid_min, self.grid_max)
        if not intersected: return []

        # Initialization
        # floating point representation of grid entry position
        start = ro + rd * max(t_entry, 0.0)
        
        # traversal constants
        step = np.sign(rd)
        delta = 1.0 / rd

        # indices of current voxel
        pos = np.clip(np.floor(start), 0, self.grid_size)
        
        tMax = (pos + step - start) / rd
        # Handle division by zero
        division_err = np.argwhere(rd == 0)
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

    
    def ray_aabb(self, rayOrigin: np.ndarray, rayDir: np.ndarray, boxMin: np.ndarray, boxMax: np.ndarray) -> tuple[bool, float]:
        """Returns whether a Ray intersects an Axis-aligned Bounding Box (AABB)
        and the time of intersection"""
        t1 = (boxMin[0] - rayOrigin[0]) / rayDir[0]
        t2 = (boxMax[0] - rayOrigin[0]) / rayDir[0]
        
        tmin = min(t1, t2)
        tmax = max(t1, t2)
        
        for axis in range(1, rayOrigin.size):
            t1 = (boxMin[axis] - rayOrigin[axis]) / rayDir[axis]
            t2 = (boxMax[axis] - rayOrigin[axis]) / rayDir[axis]

            # Modified from original behavior to handle NaNs
            tmin = max(tmin, min(min(t1, t2), tmax))
            tmax = min(tmax, max(max(t1, t2), tmin))

        return tmax > max(tmin, 0.0), tmin

    def normalize(self, vector: np.ndarray) -> np.ndarray:
        """Returns the normalized vector"""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm