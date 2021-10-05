from typing import Sequence

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np

from ..game.wireframe import Wireframe
from .parameters import Environment, FuelArray


class Terrain(Wireframe):
    def __init__(self, environment: Environment,
                 fuel_arrays: Sequence[Sequence[FuelArray]]) -> None:
        super().__init__()
        self.environment = environment
        self.fuel_arrays = fuel_arrays
        self.terrain_shape = (len(self.fuel_arrays), len(self.fuel_arrays[0]))
        self.create_nodes_and_edges()
    
    def create_nodes_and_edges(self) -> None:
        for i in range(self.terrain_shape[0]):
            for j in range(self.terrain_shape[1]):
                arr = self.fuel_arrays[i][j]
                nodes = [(x,y,z) for x in (i, i+1) for y in (j, j+1) for z in (arr.tile.z, arr.tile.z)]
                # Compute offset for node indices
                offset = 8*i + j
                edges = [(n+offset,n+4+offset) for n in range(0,4)]+[(n+offset,n+1+offset) for n in range(0,8,2)]+[(n+offset,n+2+offset) for n in (0,1,4,5)]
                self.addNodes(nodes)
                self.addEdges(edges)



# class Terrain():
#     '''
#     This class contains environment data and a collection of FuelArrays, and
#     represents some terrain in the world. The collection of FuelArrays are
#     connected based on their loactions in the collection.
#     '''
#     def __init__(self, environment: Environment,
#                  fuel_arrays: Sequence[Sequence[FuelArray]]) -> None:
#         '''
#         Initialize some terrain given the environment and fuel array.
#         '''
#         self.environment = environment
#         self.fuel_arrays = fuel_arrays
#         # Shape of terrain in (x, y) width
#         self.terrain_shape = (len(self.fuel_arrays), len(self.fuel_arrays[0]))

#     def draw(self) -> None:
#         '''
#         Draw a 3D view of the terrain. The X and Y coordinates are obatined from
#         the indexing of the fuel_arrays sequences. The Z coordinates are
#         obtained from the Tiles themselves. 

#         Arguments:
#             None
#         Returns:
#             None
#         '''
#         @np.vectorize
#         def get_z(x, y) -> float:
#             return self.fuel_arrays[y][x].tile.z
#         x = np.arange(self.terrain_shape[0])
#         y = np.arange(self.terrain_shape[1])
#         X, Y = np.meshgrid(x, y)
#         Z = get_z(X, Y)
#         fig = plt.figure()
#         ax = plt.axes(projection='3d')
#         ax.plot_surface(X, Y, Z)
#         plt.show()
