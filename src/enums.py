"""
Enums
=====

Contains many enumeration classes for use throughout `rothermel_model` that depict pixel
burn status, the ordering of sprite layers, how much to attenuate the rate of spread on
different types of control lines, and the current game status.
"""
from dataclasses import dataclass
from enum import Enum, IntEnum, auto
from importlib import resources
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

with resources.path("assets.textures", "terrain.jpg") as path:
    TERRAIN_TEXTURE_PATH: Path = path

from .world.presets import (
    Brush,
    Chaparral,
    ClosedShortNeedleTimberLitter,
    DormantBrushHardwoodSlash,
    GrassTimberShrubOverstory,
    HardwoodLongNeedlePineTimber,
    HeavyLoggingSlash,
    LightLoggingSlash,
    MediumLoggingSlash,
    NBAgriculture,
    NBBarren,
    NBNoData,
    NBSnowIce,
    NBUrban,
    NBWater,
    ShortGrass,
    SouthernRough,
    TallGrass,
    TimberLitterUnderstory,
)

DRY_TERRAIN_BROWN_IMG: Image.Image = Image.fromarray(
    np.full((10, 10, 3), (205, 133, 63), dtype=np.uint8)
)

BURNED_RGB_COLOR: Tuple[int, int, int] = (139, 69, 19)


class BurnStatus(IntEnum):
    """The status of each pixel in a `fire_map`

    Current statuses are:
        - UNBURNED
        - BURNING
        - BURNED
        - FIRELINE
        - SCRATCHLINE
        - WETLINE
    """

    UNBURNED: int = 0
    BURNING: int = auto()
    BURNED: int = auto()
    FIRELINE: int = auto()
    SCRATCHLINE: int = auto()
    WETLINE: int = auto()


@dataclass
class RoSAttenuation:
    """The factor by which to attenuate the rate of spread (RoS), based on control line
    type

    The only classes that are attenuated are the different control lines:
        - FIRELINE
        - SCRATCHLINE
        - WETLINE
    """

    FIRELINE: float = 980
    SCRATCHLINE: float = 490
    WETLINE: float = 245


class SpriteLayer(IntEnum):
    """The types of layers for sprites

    This determines the order with which sprites are layered and displayed on top of each
    other. The higher the number, the closer to the top of the layer stack. From bottom
    to top:
        - TERRAIN
        - FIRE
        - LINE
        - RESOURCE
    """

    TERRAIN: int = 1
    FIRE: int = 2
    LINE: int = 3
    RESOURCE: int = 4


class GameStatus(Enum):
    """The different statuses that the game can be in

    Currently it can only be in the following modes:
        - QUIT
        - RUNNING
    """

    QUIT = auto()
    RUNNING = auto()


@dataclass
class FuelConstants:
    """
    Constants to be used in the file and can be imported for reference.

    Parameters:
        W_0_MIN: Oven-dry Fuel Load (lb/ft^2) minimum.
        W_0_MAX: Oven-dry Fuel Load (lb/ft^2) maximum.
        DELTA: Fuel bed depth (ft) min and max.
        M_X: Dead fuel moisture of extinction min and max.
        SIGMA: Surface-area-to-volume ratio (ft^2/ft^3) min and max.
    """

    W_0_MIN: float = 0.2
    W_0_MAX: float = 0.6
    DELTA: float = 6.000
    M_X: float = 0.2000
    SIGMA: int = 1739


@dataclass
class ElevationConstants:
    """
    Constants to be used in the file and can be imported for reference.

    Paremeters:
        MIN_ELEVATION: Minimum elevation (ft). Based on the elevation of Death Valley,
                       the lowest point in California and the US in general.
        MAX_ELEVATION: Maximum elevation (ft). Based on the elevation of the treeline in
                       California.
        MEAN_ELEVATION: Mean elevation (ft). Based on the average elevation of California.
                        From [NRC.gov](https://www.nrc.gov/docs/ML1408/ML14086A640.pdf).
    """

    MIN_ELEVATION: int = -282
    MAX_ELEVATION: int = 11_000
    MEAN_ELEVATION: int = 2_500


@dataclass
class WindConstants:
    """
    Constants to be used in the file and can be imported for reference.

    Paremeters:
        MIN_SPEED: Minimum wind speed (mph).
        MAX_SPEED: Maximum wind speed (mph). The maximum recorded wind speed in CA is 209
                   mph, so to be safe, let's set it to 250. You never know with climate
                   change.
    """

    MIN_SPEED: int = 0
    MAX_SPEED: int = 250


FuelModelToFuel = {
    1: ShortGrass,
    2: GrassTimberShrubOverstory,
    3: TallGrass,
    4: Chaparral,
    5: Brush,
    6: DormantBrushHardwoodSlash,
    7: SouthernRough,
    8: ClosedShortNeedleTimberLitter,
    9: HardwoodLongNeedlePineTimber,
    10: TimberLitterUnderstory,
    11: LightLoggingSlash,
    12: MediumLoggingSlash,
    13: HeavyLoggingSlash,
    91: NBUrban,
    92: NBSnowIce,
    93: NBAgriculture,
    98: NBWater,
    99: NBBarren,
    -32768: NBNoData,
    -9999: NBNoData,
    32767: NBNoData,
}

FuelModelRGB13 = {
    1: [1.0, 1.0, 0.745098039],
    2: [1.0, 1.0, 0.0],
    3: [0.901960784, 0.77254902, 0.043137255],
    4: [1.0, 0.82745098, 0.498039216],
    5: [1.0, 0.666666667, 0.4],
    6: [0.803921569, 0.666666667, 0.4],
    7: [0.537254902, 0.439215686, 0.266666667],
    8: [0.82745098, 1.0, 0.745098039],
    9: [0.439215686, 0.658823529, 0.0],
    10: [0.149019608, 0.450980392, 0.0],
    11: [0.909803922, 0.745098039, 1.0],
    12: [0.478431373, 0.556862745, 0.960784314],
    13: [0.77254902, 0.0, 1.0],
    91: [0.517647, 0.0, 0.541176],
    92: [0.623529, 0.631373, 0.941176],
    93: [0.913725, 0.45098, 1.0],
    98: [0.0, 0.0, 1.0],
    99: [0.74902, 0.74902, 0.74902],
    -32768: [1.0, 1.0, 1.0],
    -9999: [1.0, 1.0, 1.0],
    32767: [1.0, 1.0, 1.0],
}
