import unittest
from typing import Tuple

import numpy as np

from ..graph import FireSpreadGraph
from ...enums import BurnStatus


def _create_map_and_coords(screen_size: Tuple[int, int]) -> \
        Tuple[np.ndarray, Tuple[int, int], Tuple[int, int, int], Tuple[int, int, int]]:
    '''
    Create a dummy fire map and coordinates to add graph edges.
    The coordinates are always left small so that they will work with
    smaller screen_size parameters just in case it is changed.

    Arguments:
        screen_size: The size of the simulation in pixels

    Returns:
        A numpy array representing a fire map with the location of (1, 1) burning
        The location in (x, y) that is burning
        Coordinates in the x-dimension adjacent to (1, 1) that are now burning
        Coordinates in the y-dimension adjacent to (1, 1) that are now burning
    '''
    fire_map = np.full(screen_size, BurnStatus.UNBURNED)
    burning_loc = (1, 1)
    fire_map[burning_loc] = BurnStatus.BURNING
    x_coords = (0, 1, 2)
    y_coords = (0, 0, 2)

    return fire_map, burning_loc, x_coords, y_coords


class TestFireSpreadGraph(unittest.TestCase):
    '''
    Test that FireSpreadGraph works properly.
    '''
    def setUp(self) -> None:
        self.screen_size = (8, 8)
        self.fs_graph = FireSpreadGraph(self.screen_size)
        return super().setUp()

    def test___init__(self) -> None:
        '''
        Test that the graph is initialized with the correct number of nodes.
        '''

        num_nodes_in_graph = len(self.fs_graph.graph.nodes)
        true_num_nodes = self.screen_size[0] * self.screen_size[1]
        self.assertEqual(num_nodes_in_graph,
                         true_num_nodes,
                         msg='The number of nodes created and in the graph are '
                         f'{num_nodes_in_graph}, but should be {true_num_nodes}')

    def test_add_vertices_from_manager(self) -> None:
        '''
        Test that the new new burning locations passed from the FireManager
        can be used to create vertices.
        '''
        fire_map, burning_loc, x_coords, y_coords = _create_map_and_coords(
            self.screen_size)

        self.fs_graph.add_edges_from_manager(x_coords, y_coords, fire_map)

        edges = list(self.fs_graph.graph.edges)
        true_edges = [(burning_loc, (x, y)) for x, y in zip(x_coords, y_coords)]
        self.assertListEqual(edges,
                             true_edges,
                             msg=f'The created edges {edges} do not match what the '
                             f'true edges should be {true_edges}')

    def test_draw(self) -> None:
        '''
        Test that the draw method works with and without a background image.
        '''
        fire_map, burning_loc, x_coords, y_coords = _create_map_and_coords(
            self.screen_size)

        self.fs_graph.add_edges_from_manager(x_coords, y_coords, fire_map)

        # with self.subTest('With background'):
        #     bg_image = np.full(self.screen_size + (3, ), 127)
        #     fig = self.fs_graph.draw(bg_image)
        #     return
