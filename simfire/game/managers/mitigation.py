from typing import List, Optional, Sequence, Tuple, Type, Union

import numpy as np

from ...enums import BurnStatus
from ..sprites import FireLine, ScratchLine, Terrain, WetLine

PointType = Tuple[int, int]
PointsType = Sequence[PointType]


class ControlLineManager:
    """
    Base class to create and manage control lines and allow for the creation of more
    control lines while the game is running. Child classes will change the `line_type`,
    `sprite_type`, and add the initial points with `

    Call `update()` to add points.
    """

    def __init__(
        self,
        size: int,
        pixel_scale: float,
        terrain: Terrain,
        headless: bool = False,
    ) -> None:
        """
        Initialize the class with the display size of each `ControlLine` sprite,
        the `pixel_scale`, and the `Terrain` that the `ControlLine`s will be placed.

        Arguments:
            size: The display size of each `ControlLine` point.
            pixel_scale: The amount of ft each pixel represents. This is needed
                         to track how much a fire has burned at a certain
                         location since it may take more than one update for
                         a pixel/location to catch on fire depending on the
                         rate of spread.
            terrain: The Terrain that describes the simulation/game
            headless: Flag to run in a headless state. This will allow PyGame objects to
                      not be initialized.
        """
        self.size = size
        self.pixel_scale = pixel_scale
        self.terrain = terrain
        self.line_type: BurnStatus
        self.sprite_type: Union[Type[FireLine], Type[ScratchLine], Type[WetLine]]
        # The child classes will instantiate self.sprites with the correct typing
        # (e.g. List[FireLine])
        self.sprites: List
        self.headless = headless

    def _add_point(self, point: PointType) -> None:
        """
        Updates self.sprites to add a new point to the control line
        """
        new_sprite = self.sprite_type(point, self.size, self.headless)
        self.sprites.append(new_sprite)

    def update(
        self, fire_map: np.ndarray, points: Optional[PointsType] = None
    ) -> np.ndarray:
        """
        Updates the passed in `fire_map` with new `ControlLine` `points`.

        Arguments:
            fire_map: The `fire_map` to update with new points

        Returns:
            fire_map: The upadated fire map with the control lines added.
        """
        if points is None:
            pass
        else:
            for point in points:
                x, y = point
                fire_map[y, x] = self.line_type
                self._add_point(point)

        return fire_map


class FireLineManager(ControlLineManager):
    """
    Manages the placement of `FireLines` and `FireLine` sprites. Should have varying
    physical characteristics from `ScratchLines` and `WetLines`.

    Call `update()` to add points.
    """

    def __init__(
        self, size: int, pixel_scale: float, terrain: Terrain, headless: bool = False
    ) -> None:
        """
        Initialize the class with the display size of each `FireLine` sprite,
        the `pixel_scale`, and the `Terrain` that the `FireLine`s will be placed.

        Sets the `line_type` to `BurnStatus.FIRELINE`.

        Arguments:
            size: The display size of each `FireLine` point.
            pixel_scale: The amount of ft each pixel represents. This is needed
                         to track how much a fire has burned at a certain
                         location since it may take more than one update for
                         a pixel/location to catch on fire depending on the
                         rate of spread.
            terrain: The Terrain that describes the simulation/game
            headless: Flag to run in a headless state. This will allow PyGame objects to
                      not be initialized.
        """
        super().__init__(
            size=size,
            pixel_scale=pixel_scale,
            terrain=terrain,
            headless=headless,
        )
        self.line_type = BurnStatus.FIRELINE
        self.sprite_type = FireLine
        self.sprites: List[FireLine] = []


class ScratchLineManager(ControlLineManager):
    """
    Manages the placement of `FireLines` and `ScratchLine` sprites. Should have varying
    physical characteristics from `FireLines` and `WetLines`.

    Call `update()` to add points.
    """

    def __init__(
        self, size: int, pixel_scale: float, terrain: Terrain, headless: bool = False
    ) -> None:
        """
        Initialize the class with the display size of each `ScratchLine` sprite,
        the `pixel_scale`, and the `Terrain` that the `ScratchLine`s will be placed.

        Sets the `line_type` to `BurnStatus.SCRATCHLINE`.

        Arguments:
            size: The display size of each `ScratchLine` point.
            pixel_scale: The amount of ft each pixel represents. This is needed
                         to track how much a fire has burned at a certain
                         location since it may take more than one update for
                         a pixel/location to catch on fire depending on the
                         rate of spread.
            terrain: The Terrain that describes the simulation/game
            points: The list of all ((x1, y1), (x2, y2)) pairs of pairs that designate
                    between which two points control lines will be drawn.
        """
        super().__init__(
            size=size, pixel_scale=pixel_scale, terrain=terrain, headless=headless
        )
        self.line_type = BurnStatus.SCRATCHLINE
        self.sprite_type = ScratchLine
        self.sprites: List[ScratchLine] = []


class WetLineManager(ControlLineManager):
    """
    Manages the placement of `WetLines` and `WetLine` sprites. Should have varying
    physical characteristics from `ScratchLines` and `FireLines`.

    Call `update()` to add points.
    """

    def __init__(
        self, size: int, pixel_scale: float, terrain: Terrain, headless: bool = False
    ) -> None:
        """
        Initialize the class with the display size of each `WetLine` sprite,
        the `pixel_scale`, and the `Terrain` that the `WetLine`s will be placed.

        Sets the `line_type` to `BurnStatus.WETLINE`.

        Arguments:
            size: The display size of each `WetLine` point.
            pixel_scale: The amount of ft each pixel represents. This is needed
                         to track how much a fire has burned at a certain
                         location since it may take more than one update for
                         a pixel/location to catch on fire depending on the
                         rate of spread.
            terrain: The Terrain that describes the simulation/game
            points: The list of all ((x1, y1), (x2, y2)) pairs of pairs that designate
                    between which two points control lines will be drawn.
        """
        super().__init__(
            size=size, pixel_scale=pixel_scale, terrain=terrain, headless=headless
        )
        self.line_type = BurnStatus.WETLINE
        self.sprite_type = WetLine
        self.sprites: List[WetLine] = []
