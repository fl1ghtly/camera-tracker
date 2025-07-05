import pyvista as pv
import numpy as np
from ray import Ray

SHOW_GRID = False
SHOW_RAY = False
SHOW_TOP_PERCENTILE = False
POINT_SIZE = 12.

class Graph:
    plotter: pv.Plotter
    
    def __init__(self) -> None:
        self.plotter = pv.Plotter()
        
    def show(self) -> None:
        self.plotter.show_grid() # type: ignore
        self.plotter.show(interactive_update=True)
        
    def update(self) -> None:
        self.plotter.update()
        
    def add_voxels(self, voxels: np.ndarray, origin: np.ndarray, size: float) -> None:
        if SHOW_GRID:
            self._create_grid(voxels, origin, size)
        else:
            self._create_point_cloud(voxels, origin, size)
    
    def add_ray(self, ray: Ray, color: str, reversed=False) -> None:
        if not SHOW_RAY: return
        rev = -1 if reversed else 1
        line = pv.Line(ray.origin, ray.origin + ray.norm_dir * 300 * rev)
        self.plotter.add_mesh(line, color=color, line_width=2)      
        
    def extract_percentile_index(self, data: np.ndarray, percentile: float):
        """Returns the indices of all data points above a certain percentile"""
        p = np.percentile(data[data != 0], percentile)
        return np.nonzero(data >= p)
    
    def _create_point_cloud(self, voxels: np.ndarray, origin: np.ndarray, size: float):
        # Points are the (x, y, z) of the center of each voxel
        voxel_center = np.full(3, size / 2)
        if SHOW_TOP_PERCENTILE:
            ind = self.extract_percentile_index(voxels, 99.9)
        else:
            ind = np.nonzero(voxels)
        points = np.transpose(ind) * size + voxel_center + origin

        if len(points) <= 0:
            return

        cloud = pv.PolyData(points)
        cloud['Values'] = voxels[ind]
        
        self.plotter.add_points(cloud, 
                                render_points_as_spheres=True,
                                # opacity='geom',
                                point_size=POINT_SIZE,
                                name="point_cloud")
    
    def _create_grid(self, voxels: np.ndarray, origin: np.ndarray, size: float):
        grid = pv.ImageData()
        grid.dimensions = np.array(voxels.shape) + 1
        grid.spacing = (size, size, size)
        grid.origin = origin
        grid.cell_data['Values'] = voxels.flatten(order="F")

        self.plotter.add_mesh(grid, show_edges=True)