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

from .world.presets import (ShortGrass, GrassTimberShrubOverstory, TallGrass, Chaparral,
                            Brush, DormantBrushHardwoodSlash, SouthernRough,
                            ClosedShortNeedleTimberLitter, HardwoodLongNeedlePineTimber,
                            TimberLitterUnderstory, LightLoggingSlash, MediumLoggingSlash,
                            HeavyLoggingSlash, NBUrban, NBSnowIce, NBAgriculture,
                            NBBarren, NBWater, NBNoData)

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
    32767: NBNoData
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
    32767: [1.0, 1.0, 1.0]
}
