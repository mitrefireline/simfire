from typing import List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import lines as mlines

from ..enums import BurnStatus


class FireSpreadGraph:
    """
    Class that stores the direction of fire spread from pixel to pixel.
    Each pixel is a node in the graph, with pixels/nodes connected by directed
    vertices based on the fire spread in that direction.
    """

    def __init__(self, screen_size: Tuple[int, int]) -> None:
        """
        Store the screen size and initialize a graph with a node for each pixel.
        Each node is referenced by its (x, y) location on the screen.
        The graph will have no vertices to start.

        Arguments:
            screen_size: The size of the simulation in pixels

        Returns:
            None
        """
        self.screen_size = screen_size
        self.graph = nx.DiGraph()

        self.nodes = self._create_nodes()
        self.graph.add_nodes_from(self.nodes)

    def _create_nodes(self) -> Tuple[Tuple[int, int], ...]:
        """
        Create the nodes for the graph. The nodes are tuples in (x, y) format.

        Arguments:
            None

        Returns:
            A tuple all the (x, y) nodes needed for the graph
        """
        # Each (x, y) pixel point is a node
        nodes = tuple(
            (x, y) for x in range(self.screen_size[1]) for y in range(self.screen_size[0])
        )

        return nodes

    def get_descendant_heatmap(self, flat: bool = False) -> np.ndarray:
        """
        Create a heatmap array showing which nodes have the most descendants.
        This will show which nodes cause the most spread (but beware that nodes
        close to the starting location will inherently be more impactful.
        The heatmap can be flat for use with self.draw(), or reshaped for creating
        an image that aligns with the game screen.

        Arguments:
            flat: Flag indicating whether the returned value should remain as a
                  flat array where each index in the array aligns with the node
                  in self.graph.nodes, or the returned value should be reshaped
                  to represent an image using the (x, y) coordinates of the nodes

        Returns:
            A numpy array of shape (len(self.graph.nodes),) if flat==True
            A numpy array of shape (Y, X), where Y is the largest y-coordinate
            in self.nodes, and X is the largest x-coordinate in self.nodes
        """
        heatmap: Union[List[int], List[List[int]]]
        if flat:
            heatmap = [len(nx.descendants(self.graph, node)) for node in self.graph.nodes]
        else:
            yrange, xrange = self.screen_size
            heatmap = [
                [len(nx.descendants(self.graph, (y, x))) for x in range(xrange)]
                for y in range(yrange)
            ]

        return np.array(heatmap, dtype=np.uint8)

    def add_edges_from_manager(
        self,
        x_coords: Union[int, Sequence[int]],
        y_coords: Union[int, Sequence[int]],
        fire_map: np.ndarray,
    ) -> None:
        """
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
        """
        # Singular numpy arrays called with arr.tolist() are returned as single
        # values. Convert to list for compatibility
        if isinstance(x_coords, int) and isinstance(y_coords, int):
            x_coords = [x_coords]
            y_coords = [y_coords]
        elif isinstance(x_coords, Sequence) and isinstance(y_coords, Sequence):
            x_coords = list(x_coords)
            y_coords = list(y_coords)
        else:
            raise ValueError(
                "x_coords and y_coords should both be int or Sequence. "
                f"Got {type(x_coords)} and {type(y_coords)}, respectively"
            )
        # Check that the x and y coordinates will align
        if (x_len := len(x_coords)) != (y_len := len(y_coords)):
            raise ValueError(
                f"The length of x_coords ({x_len}) should match "
                f"the length of y_coords ({y_len}"
            )

        for x, y in zip(x_coords, y_coords):
            adj_locs = (
                (x + 1, y),
                (x + 1, y + 1),
                (x, y + 1),
                (x - 1, y + 1),
                (x - 1, y),
                (x - 1, y - 1),
                (x, y - 1),
                (x + 1, y - 1),
            )
            # Filter any locations that are outside of the screen area or not
            # currently burning
            adj_locs = tuple(
                filter(
                    lambda xy: xy[0] < fire_map.shape[1]
                    and xy[0] >= 0
                    and xy[1] < fire_map.shape[0]
                    and xy[1] >= 0
                    and fire_map[xy[1], xy[0]] == BurnStatus.BURNING,
                    adj_locs,
                )
            )
            # Create the edges by connecing the adjacent locations/nodes to the
            # current node
            edges = [(adj_loc, (x, y)) for adj_loc in adj_locs]
            self.graph.add_edges_from(edges)

    def draw(
        self,
        background_image: Optional[np.ndarray] = None,
        show_longest_path: bool = True,
        use_heatmap: bool = True,
    ) -> plt.Figure:
        """
        Draw the graph with the nodes/pixels in the correct locations and the
        edges shown as arrows connecting the nodes/pixels.

        Arguemnts:
            background_image: A numpy array containing the background image on
                              which to overlay the graph. If not specified,
                              then no background image will be used
            show_longest_path: Flag to draw/highlight the longest path in the graph
            use_heatmap: Flag to color the nodes using a heatmap based on
                            node descendants

        Returns:
            A matplotlib.pyplot.Figure of the drawn graph
        """
        # TODO: This still doesn't quite seem to line up the image and graph
        # Might need to manually draw_eges and draw_nodes
        pos = {(x, y): (x, y) for (x, y) in self.nodes}
        fig, ax = plt.subplots(1, 1, figsize=(12.8, 9.6))
        fig.tight_layout()
        if background_image is not None:
            ax.imshow(background_image)

        # Initialize list for potential legend elements
        legend_elements = []
        # Get figure facecolor for use with legened element "backgrounds"
        facecolor = fig.get_facecolor()

        # Scale the node and marker size based on the figure size
        fig_size_pixels = fig.dpi * fig.get_size_inches().mean()
        node_size = 0.002 * fig_size_pixels
        markersize = 0.01 * fig_size_pixels

        # Create a legend element for the edges (fire paths)
        edge_path_artist = mlines.Line2D(
            [0],
            [0],
            color=facecolor,
            marker=">",
            markerfacecolor="k",
            markersize=markersize,
            label="Fire Spread Path",
        )
        legend_elements.append(edge_path_artist)

        # All edges will be black by default. Color the edges of the longest path in
        # red if specified
        if show_longest_path:
            longest_path = nx.dag_longest_path(self.graph)
            longest_edges = [
                (longest_path[i], longest_path[i + 1])
                for i in range(len(longest_path) - 1)
            ]
            edge_color = [
                "r" if edge in longest_edges else "k" for edge in self.graph.edges
            ]
            # Create artist to add to legend
            longest_path_artist = mlines.Line2D(
                [0],
                [0],
                color=facecolor,
                marker=">",
                markerfacecolor="r",
                markersize=markersize,
                label="Longest Fire Spread Path",
            )
            legend_elements.append(longest_path_artist)
        else:
            # 'k' is black for matplotlib
            edge_color = ["k" for edge in self.graph.edges]

        # All nodes with outbound edges will red by default.
        # 'r' is red for matplotlib
        node_color = [
            "r" if self.graph.out_degree(node) > 0 else (0, 0, 0, 0)
            for node in self.graph.nodes
        ]

        # If a heatmap is used, the node sizes will be scaled based on the
        # heatmap values
        if use_heatmap:
            node_heatmap = self.get_descendant_heatmap(flat=True)
            node_heatmap = node_heatmap / node_heatmap.max()
            node_size = [50**val * node_size for val in node_heatmap]
            node_size_artist = mlines.Line2D(
                [0],
                [0],
                color=facecolor,
                marker="o",
                markerfacecolor="r",
                markersize=markersize,
                label="Fire Node " "(larger means more descendants)",
            )
        else:
            node_size_artist = mlines.Line2D(
                [0],
                [0],
                color=facecolor,
                marker="o",
                markerfacecolor="r",
                markersize=markersize,
                label="Fire Node",
            )

        legend_elements.append(node_size_artist)

        nx.draw_networkx(
            self.graph,
            pos=pos,
            ax=ax,
            node_size=node_size,
            node_color=node_color,
            with_labels=False,
            arrowstyle="->",
            edge_color=edge_color,
        )

        ax.legend(handles=legend_elements, loc="lower right")

        return fig
