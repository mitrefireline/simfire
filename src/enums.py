'''
Enums
=====

Contains many enumeration classes for use throughout `rothermel_model` that depict pixel
burn status, the ordering of sprite layers, how much to attenuate the rate of spread on
different types of control lines, and the current game status.
'''
from dataclasses import dataclass
from enum import auto, Enum, IntEnum

import numpy as np
from PIL import Image

from .config import terrain_size

TERRAIN_TEXTURE_PATH: str = 'assets/textures/terrain.jpg'
FIRE_TEXTURE_PATH: str = 'assets/textures/flames.png'
FIRELINE_TEXTURE_PATH: str = 'assets/textures/fire_line.jpg'
SCRATCHLINE_TEXTURE_PATH: str = 'assets/textures/scratch_line.jpg'
WETLINE_TEXTURE_PATH: str = 'assets/textures/wet_line.jpg'

DRY_TERRAIN_BROWN_IMG: Image.Image = Image.fromarray(
    np.full((terrain_size, terrain_size, 3), (205, 133, 63), dtype=np.uint8))

BURNED_RGB_COLOR = (139, 69, 19)


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
    FIRELINE: float = 0.01
    SCRATCHLINE: float = 0.02
    WETLINE: float = 0.03


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
