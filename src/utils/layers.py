import numpy as np

from ..world.elevation_functions import ElevationFn


class DataLayer():
    '''
    Layer class that allows for simulation data layers to be stored and used. A layer
    represents data at every pixel/point in the simulation area.
    '''
    def __init__(self, height: int, width: int) -> None:
        '''
        Initialize the data layer with None values to avoid errors when no data is loaded.

        Arguments:
            height: The height of the data layer
            width: The width of the data layer

        Returns:
            None
        '''
        self.height = height
        self.width = width
        self.data = np.full((height, width), None, dtype=np.float)


class ElevationLayer(DataLayer):
    '''
    Layer that stores elevation data computed from a function.
    '''
    def __init__(self, height, width, elevation_fn: ElevationFn) -> None:
        '''
        Initialize the elvation layer by computing the elevations and contours.

        Arguments:
            height: The height of the data layer
            width: The width of the data layer
            elevation_fn: A callable function that converts (x, y) coorindates to
                          elevations.
        '''
        super().__init__(height, width)
        self.elevation_fn = elevation_fn
        self.data = self._make_contour_and_data()

    def _make_data(self) -> np.ndarray:
        '''
        Use self.elevation_fn to make the elevation data layer.

        Arguments:
            None

        Returns:
            A numpy array containing the elevation data
        '''
        x = np.arange(self.width)
        y = np.arange(self.height)
        X, Y = np.meshgrid(x, y)
        elevation_fn_vect = np.vectorize(self.elevation_fn)
        elevations = elevation_fn_vect(X, Y)

        return elevations
