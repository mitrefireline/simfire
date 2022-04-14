from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
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
        # Each (x, y) pixel point is a node
        nodes = tuple((x, y) for x in range(self.screen_size[1])
                      for y in range(self.screen_size[0]))

        return nodes

    def _create_heatmap(self, xrange: int, yrange: int) -> np.ndarray:
        '''
        Create a heatmap that shows how many child nodes each node is attached to.

        Arguments:
            None

        Returns:
            The created heatmap as a numpy array
        '''
        # Initialize dictionary for quick lookup
        # Keys are nodes, values are lists of all children
        children_dict = {}

        def compute_num_children(node: Tuple[int, int]) -> List[Tuple[int, int]]:
            '''
            Recursive function that computes how many children a node has (i.e. the
            total number of nodes that can be reached from this node).

            Arguments:
                node: The node to compute a score for

            Returns:
                A list of all child nodes
            '''
            children = set()
            # Get the out-edges from the node
            out_edges = self.graph.edges(node)
            # Get the connected nodes based on the edges
            conn_nodes = [edge[1] for edge in out_edges]
            for conn_node in conn_nodes:
                if conn_node in children_dict:
                    children.update(children_dict[conn_node])
                    continue
                else:
                    if self.graph.out_degree(conn_node) == 0:
                        return [node]
                    else:
                        children.update(compute_num_children(conn_node))
            # Add the result to the lookup dictionary
            children_dict[node] = children
            return list(children)

        heatmap = [[len(compute_num_children((y, x))) for y in range(yrange)]
                   for x in range(xrange)]
        heatmap = np.array(heatmap)

        return heatmap

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
        # Singular numpy arrays called with arr.tolist() are returned as single
        # values. Convert to list for compatibility
        if isinstance(x_coords, int):
            x_coords = [x_coords]
            y_coords = [y_coords]
        # Check that the x and y coordinates will align
        if (x_len := len(x_coords)) != (y_len := len(y_coords)):
            raise ValueError(f'The length of x_coords ({x_len}) should match '
                             f'the length of y_coords ({y_len}')

        for x, y in zip(x_coords, y_coords):
            adj_locs = ((x + 1, y), (x + 1, y + 1), (x, y + 1), (x - 1, y + 1),
                        (x - 1, y), (x - 1, y - 1), (x, y - 1), (x + 1, y - 1))
            # Filter any locations that are outside of the screen area or not
            # currently burning
            adj_locs = tuple(
                filter(
                    lambda xy: xy[0] < fire_map.shape[1] and xy[0] >= 0 and xy[
                        1] < fire_map.shape[0] and xy[1] >= 0 and fire_map[xy[1], xy[0]]
                    == BurnStatus.BURNING, adj_locs))
            # Create the edges by connecing the adjacent locations/nodes to the
            # current node
            edges = [(adj_loc, (x, y)) for adj_loc in adj_locs]
            self.graph.add_edges_from(edges)

    def draw(self,
             background_image: np.ndarray = None,
             show_longest_path: bool = True,
             create_heatmap: bool = True) -> plt.Figure:
        '''
        Draw the graph with the nodes/pixels in the correct locations and the
        edges shown as arrows connecting the nodes/pixels.

        Arguemnts:
            background_image: A numpy array containing the background image on
                              which to overlay the graph. If not specified,
                              then no background image will be used
            show_longest_path: Flag to draw/highlight the longest path in the graph
            create_heatmap: Flag to create a heatmap based on node outbound connectivity

        Returns:
            A matplotlib.pyplot.Figure of the drawn graph
        '''
        # TODO: This still doesn't quite seem to line up the image and graph
        # Might need to manually draw_eges and draw_nodes
        pos = {(x, y): (x, y) for (x, y) in self.nodes}
        fig, ax = plt.subplots(1, 1)
        fig.tight_layout()
        if background_image is not None:
            ax.imshow(background_image)

        if show_longest_path:
            longest_path = nx.dag_longest_path(self.graph)
            longest_edges = [(longest_path[i], longest_path[i + 1])
                             for i in range(len(longest_path) - 1)]
            edge_color = [
                'r' if edge in longest_edges else 'k' for edge in self.graph.edges
            ]
        else:
            edge_color = 'k'

        if create_heatmap:
            # Need to compute out_degree for all nodes
            yrange, xrange = background_image.shape[:2]
            heatmap = self._create_heatmap(xrange, yrange)
            # heatmap = [[self.graph.out_degree((y, x)) for y in range(yrange)]
            #            for x in range(xrange)]

        nx.draw_networkx(self.graph,
                         pos=pos,
                         ax=ax,
                         node_size=0,
                         with_labels=False,
                         arrowstyle='->',
                         edge_color=edge_color)

        return fig
