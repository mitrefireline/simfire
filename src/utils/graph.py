from typing import Sequence, Tuple

import networkx as nx
import numpy as np

from ..enums import BurnStatus


class FireSpreadGraph():
    '''
    Class that stores the direction of fire spread from pixel to pixel.
    Each pixel is a node in the graph, with pixels/nodes connected by directed
    vertices based on the fire spread in that direction.
    '''
    def __init__(self, screen_size: Tuple[int, int]) -> None:
        '''
        Store the screen size and initialize a graph with a node for each pixel.
        Each node is referenced by its (x, y) location on the screen.
        The graph will have no vertices to start.

        Arguments:
            screen_size: The size of the simulation in pixels

        Returns:
            None
        '''
        self.screen_size = screen_size
        self.graph = nx.DiGraph()

        self.nodes = self._create_nodes()
        self.graph.add_nodes_from(self.nodes)

    def _create_nodes(self) -> Tuple[Tuple[int, int]]:
        '''
        Create the nodes for the graph. The nodes are tuples in (x, y) format.

        Arguments:
            None

        Returns:
            A tuple all the (x, y) nodes needed for the graph
        '''
        nodes = tuple((x, y) for x in range(self.screen_size[1])
                      for y in range(self.screen_size[0]))

        return nodes

    def add_edges_from_manager(self, x_coords: Sequence[int], y_coords: Sequence[int],
                               fire_map: np.ndarray) -> None:
        '''
        Update the graph to include edges to newly burning nodes/pixels in
        coordinates (x_coords[i], y_coords[i]) from any adjacent node/pixel in
        fire_map that is currently burning.

        Arguments:
            x_coords: The x coordinates of the newly burning nodes/pixels
            y_coords: The y coordinates of the newly burning nodes/pixels
            fire_map: fire_map: The numpy array that tracks the fire's burn
                                status for each pixel in the simulation

        Returns:
            None
        '''
        if (x_len := len(x_coords)) != (y_len := len(y_coords)):
            raise ValueError(f'The length of x_coords ({x_len}) should match '
                             f'the length of y_coords ({y_len}')
        for x, y in zip(x_coords, y_coords):
            adj_locs = ((x + 1, y), (x + 1, y + 1), (x, y + 1), (x - 1, y + 1),
                        (x - 1, y), (x - 1, y - 1), (x, y - 1), (x + 1, y - 1))
            for adj_loc in adj_locs:
                # If an adjacent pixel is burning, it contributes to this pixel
                if fire_map[adj_loc] == BurnStatus.BURNING:
                    # Add a connection from the currently burning node to the
                    # newly burning node
                    self.graph.add_edge(adj_loc, (x, y))

    def draw(self) -> None:
        '''
        Draw the graph with the nodes/pixels in the correct locations and the
        edges shown as arrows connecting the nodes/pixels.

        Arguemnts:
            None

        Returns:
            None
        '''
        # TODO: Write this function that returns a pyplot Figure or Axes object
        pass
