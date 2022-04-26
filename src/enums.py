'''
Enums
=====

Contains many enumeration classes for use throughout `rothermel_model` that depict pixel
burn status, the ordering of sprite layers, how much to attenuate the rate of spread on
different types of control lines, and the current game status.
'''
from typing import Tuple
from dataclasses import dataclass
from pathlib import Path
from importlib import resources

from enum import auto, Enum, IntEnum

import numpy as np
from PIL import Image

with resources.path('assets.textures', 'terrain.jpg') as path:
    TERRAIN_TEXTURE_PATH: Path = path

DRY_TERRAIN_BROWN_IMG: Image.Image = Image.fromarray(
    np.full((10, 10, 3), (205, 133, 63), dtype=np.uint8))

BURNED_RGB_COLOR: Tuple[int, int, int] = (139, 69, 19)


class BurnStatus(IntEnum):
    '''The status of each pixel in a `fire_map`

    Current statuses are:
        - UNBURNED
        - BURNING
        - BURNED
        - FIRELINE
        - SCRATCHLINE
        - WETLINE
    '''
    UNBURNED = 0
    BURNING = auto()
    BURNED = auto()
    FIRELINE = auto()
    SCRATCHLINE = auto()
    WETLINE = auto()


@dataclass
class RoSAttenuation:
    '''The factor by which to attenuate the rate of spread (RoS), based on control line
    type

    The only classes that are attenuated are the different control lines:
        - FIRELINE
        - SCRATCHLINE
        - WETLINE
    '''
    FIRELINE: float = 980
    SCRATCHLINE: float = 490
    WETLINE: float = 245


class SpriteLayer(IntEnum):
    '''The types of layers for sprites

    This determines the order with which sprites are layered and displayed on top of each
    other. The higher the number, the closer to the top of the layer stack. From bottom
    to top:
        - TERRAIN
        - FIRE
        - LINE
        - RESOURCE
    '''
    TERRAIN = 1
    FIRE = 2
    LINE = 3
    RESOURCE = 4


class GameStatus(Enum):
    '''The different statuses that the game can be in

    Currently it can only be in the following modes:
        - QUIT
        - RUNNING
    '''
    QUIT = auto()
    RUNNING = auto()


@dataclass
class FuelConstants:
    '''
    Constants to be used in the file and can be imported for reference.

    Parameters:
        W_0_MIN: Oven-dry Fuel Load (lb/ft^2) minimum.
        W_0_MAX: Oven-dry Fuel Load (lb/ft^2) maximum.
        DELTA: Fuel bed depth (ft) min and max.
        M_X: Dead fuel moisture of extinction min and max.
        SIGMA: Surface-area-to-volume ratio (ft^2/ft^3) min and max.
    '''
    W_0_MIN: float = 0.2
    W_0_MAX: float = 0.6
    DELTA: float = 6.000
    M_X: float = 0.2000
    SIGMA: int = 1739


@dataclass
class ElevationConstants:
    '''
    Constants to be used in the file and can be imported for reference.

    Paremeters:
        MIN_ELEVATION: Minimum elevation (ft). Based on the elevation of Death Valley,
                       the lowest point in California and the US in general.
        MAX_ELEVATION: Maximum elevation (ft). Based on the elevation of the treeline in
                       California.
        MEAN_ELEVATION: Mean elevation (ft). Based on the average elevation of California.
    '''
    MIN_ELEVATION: int = -282
    MAX_ELEVATION: int = 11_000
    MEAN_ELEVATION: int = 2_900
