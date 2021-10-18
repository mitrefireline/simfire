from typing import Tuple

import numpy as np
from skimage.draw import line

from ..sprites import FireLine, Terrain
from ...enums import BurnStatus

PointType = Tuple[int, int]
PointsType = Tuple[PointType, ...]


class ControlLineManager():
    '''
    Base class to create and manage control lines and allow for the creation of more
    control lines while the game is running. Child classes will
    '''
    def __init__(self,
                 size: int,
                 pixel_scale: int,
                 terrain: Terrain,
                 points: PointsType = []) -> None:
        '''
        Initialize the class by recording all initial control line locations by
        designating each of their two endpoints.

        Create the sprites for all control lines and mark the location of the initial
        control lines.

        Arguments:
            points: The list of all ((x1, y1), (x2, y2)) pairs of pairs that designate
                    between which two points control lines will be drawn.
            pixel_scale: The amount of ft each pixel represents. This is needed
                         to track how much a fire has burned at a certain
                         location since it may take more than one update for
                         a pixel/location to catch on fire depending on the
                         rate of spread.
            terrain: The Terrain that describes the simulation/game

        Returns:
            None
        '''
        self.size = size
        self.pixel_scale = pixel_scale
        self.terrain = terrain
        if len(points) == 0:
            self.points = []
        elif isinstance(points[0], int):
            self.points = (points, )
        else:
            self.points = points
        self.line_type = None
        self.sprite_type = None

    def _add_initial_points(self, points: PointsType) -> None:
        '''
        Uses `self._draw_line` to draw all initial lines passed in to `self.__init__`

        Arguments:
            None

        Returns:
            None
        '''
        sprites = []
        for point in points:
            sprites.append(self.sprite_type(point, size=self.size))

        return sprites

    def _draw_line(self, point_1: PointType, point_2: PointType) -> None:
        '''
        Updates `self.fire_map` with a control line of type `self.line_type` between the
        provided points: `point_1`, and `point_2`.

        Arguments:
            point_1: A tuple of indices for the first point that will create the control
                     line.
            point_2: A tuple of indices for the second point that will create the control
                     line.

        Returns:
            None
        '''
        rows, columns = line(point_1[1], point_1[0], point_2[1], point_2[0])
        self.fire_map[rows, columns] = self.line_type

    def _add_point(self, point: PointType) -> None:
        '''
        Updates self.sprites to add a new point to the control line
        '''
        self.sprites.append(self.sprite_type(point, self.size))

    def update(self, fire_map: np.ndarray, points: PointsType = []) -> np.ndarray:
        '''


        Arguments:
            None

        Returns:
            fire_map: The upadated fire map with the control lines added
        '''
        for point in points:
            x, y = point
            fire_map[y, x] = self.line_type
            self._add_point(point)

        return fire_map


class FireLineManager(ControlLineManager):
    '''
    '''
    def __init__(self,
                 size: int,
                 pixel_scale: int,
                 terrain: Terrain,
                 points: Tuple[Tuple[int, int], ...] = None) -> None:
        '''
        '''
        super().__init__(size=size,
                         points=points,
                         pixel_scale=pixel_scale,
                         terrain=terrain)
        self.line_type = BurnStatus.FIRELINE
        self.sprite_type = FireLine
        self.sprites = self._add_initial_points(self.points)


class ScratchLineManager(ControlLineManager):
    '''
    '''
    def __init__(self,
                 pixel_scale: int,
                 terrain: Terrain,
                 points: Tuple[Tuple[int, int], ...] = None) -> None:
        '''
        '''
        self.line_type = BurnStatus.SCRATCHLINE
        super().__init__(points=points,
                         pixel_scale=pixel_scale,
                         terrain=terrain,
                         line_type=self.line_type)


class WetLineManager(ControlLineManager):
    '''
    '''
    def __init__(self,
                 pixel_scale: int,
                 terrain: Terrain,
                 points: Tuple[Tuple[int, int], ...] = None) -> None:
        '''
        '''
        self.line_type = BurnStatus.WETLINE
        super().__init__(points=points,
                         pixel_scale=pixel_scale,
                         terrain=terrain,
                         line_type=self.line_type)
