"""
Enums
=====

Contains many enumeration classes for use throughout `rothermel_model` that depict pixel
burn status, the ordering of sprite layers, how much to attenuate the rate of spread on
different types of control lines, and the current game status.
"""
from dataclasses import dataclass
from enum import Enum, IntEnum, auto
from pathlib import Path
from typing import Tuple

import numpy as np
import pkg_resources
from PIL import Image

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

TERRAIN_TEXTURE_PATH: Path = Path(
    pkg_resources.resource_filename("simfire.utils.textures", "terrain.jpg")
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
        - AGENT
    """

    TERRAIN: int = 1
    FIRE: int = 2
    LINE: int = 3
    AGENT: int = 4


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
                        From `NRC.gov <https://www.nrc.gov/docs/ML1408/ML14086A640.pdf>`_.
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

# PRODUCTION RATES

# Chains = 66ft x 30ft

# Line Production
# {Fuel Model: Chains / person / hour}
# https://www.nifc.gov/nicc/logistics/references/Wildland%20Fire%20Incident%20Management%20Field%20Guide.pdf
# pg 111
# Line Production Rates for Initial Action by Hand Crews in Chains per Person per Hour
# Assume using Fuel Model 13
HandLineRate = {
    1: 4.0,
    2: 3.0,
    3: 0.7,
    4: 0.4,
    5: 0.7,
    6: 0.7,
    7: 0.7,
    8: 2.0,
    9: 2.0,
    10: 1.0,
    11: 1.0,
    12: 1.0,
    13: 0.4,
}

# A lookup dictionary to calculate rate of dozer line creation
# These rates are an average of reported rates/ranges

# {Dozer Type: {Fuel Mdoel: {Slope Type: {Grade: Chains/Hr}}}}
# Dozer Type 1: HEAVY. 200 min Horse Power, [D-8, D-7, JD-950]
# Dozer Type 2: MEDIUM: 100 Min Horse Power, [D-5N, D-6N, JD-750]
# Dozer Type 3: LIGHT: 50 min Horse Power, [JD-450, JD-550, D-3, D-4]

DozerType = {
    "D-8": 1,
    "D-7": 1,
    "JD-950": 1,
    "D-5": 2,
    "D-6N": 2,
    "JD-750": 2,
    "JD-450": 3,
    "JD-550": 3,
    "D-3": 3,
    "D-4": 3,
}

DozerLineRates = {
    3: {
        1: {"up": {1: 73, 2: 43, 3: 19, 4: 4}, "down": {1: 100, 2: 100, 3: 55, 4: 10}},
        2: {"up": {1: 73, 2: 43, 3: 19, 4: 4}, "down": {1: 100, 2: 100, 3: 55, 4: 10}},
        3: {"up": {1: 58, 2: 35, 3: 14, 4: 1}, "down": {1: 75, 2: 73, 3: 33, 4: 0}},
        4: {"up": {1: 28, 2: 15, 3: 5, 4: 0}, "down": {1: 38, 2: 33, 3: 13, 4: 0}},
        5: {"up": {1: 58, 2: 35, 3: 14, 4: 1}, "down": {1: 75, 2: 73, 3: 33, 4: 0}},
        6: {"up": {1: 45, 2: 50, 3: 7, 4: 0}, "down": {1: 58, 2: 50, 3: 20, 4: 0}},
        7: {"up": {1: 45, 2: 50, 3: 7, 4: 0}, "down": {1: 58, 2: 50, 3: 20, 4: 0}},
        8: {"up": {1: 58, 2: 35, 3: 14, 4: 1}, "down": {1: 75, 2: 73, 3: 33, 4: 0}},
        9: {"up": {1: 45, 2: 50, 3: 7, 4: 0}, "down": {1: 58, 2: 50, 3: 20, 4: 0}},
        10: {"up": {1: 12, 2: 6, 3: 2, 4: 0}, "down": {1: 13, 2: 8, 3: 3, 4: 0}},
        11: {"up": {1: 20, 2: 11, 3: 4, 4: 0}, "down": {1: 28, 2: 15, 3: 5, 4: 0}},
        12: {"up": {1: 20, 2: 11, 3: 4, 4: 0}, "down": {1: 28, 2: 15, 3: 5, 4: 0}},
        13: {"up": {1: 12, 2: 6, 3: 2, 4: 0}, "down": {1: 13, 2: 8, 3: 3, 4: 0}},
    },
    2: {
        1: {"up": {1: 105, 2: 73, 3: 45, 4: 15}, "down": {1: 135, 2: 138, 3: 103, 4: 38}},
        2: {"up": {1: 105, 2: 73, 3: 45, 4: 15}, "down": {1: 135, 2: 138, 3: 103, 4: 38}},
        3: {"up": {1: 88, 2: 58, 3: 30, 4: 8}, "down": {1: 113, 2: 113, 3: 80, 4: 28}},
        4: {"up": {1: 48, 2: 28, 3: 11, 4: 1}, "down": {1: 68, 2: 71, 3: 43, 4: 10}},
        5: {"up": {1: 88, 2: 58, 3: 30, 4: 8}, "down": {1: 113, 2: 113, 3: 80, 4: 28}},
        6: {"up": {1: 68, 2: 40, 3: 19, 4: 4}, "down": {1: 93, 2: 93, 3: 63, 4: 20}},
        7: {"up": {1: 68, 2: 40, 3: 19, 4: 4}, "down": {1: 93, 2: 93, 3: 63, 4: 20}},
        8: {"up": {1: 88, 2: 58, 3: 30, 4: 8}, "down": {1: 113, 2: 113, 3: 80, 4: 28}},
        9: {"up": {1: 68, 2: 40, 3: 19, 4: 4}, "down": {1: 93, 2: 93, 3: 63, 4: 20}},
        10: {"up": {1: 15, 2: 9, 3: 4, 4: 0}, "down": {1: 23, 2: 23, 3: 10, 4: 0}},
        11: {"up": {1: 33, 2: 20, 3: 8, 4: 1}, "down": {1: 48, 2: 50, 3: 23, 4: 0}},
        12: {"up": {1: 33, 2: 20, 3: 8, 4: 1}, "down": {1: 48, 2: 50, 3: 23, 4: 0}},
        13: {"up": {1: 15, 2: 9, 3: 4, 4: 0}, "down": {1: 23, 2: 23, 3: 10, 4: 0}},
    },
    1: {
        1: {"up": {1: 120, 2: 85, 3: 53, 4: 18}, "down": {1: 148, 2: 148, 3: 113, 4: 43}},
        2: {"up": {1: 120, 2: 85, 3: 53, 4: 18}, "down": {1: 148, 2: 148, 3: 113, 4: 43}},
        3: {"up": {1: 93, 2: 63, 3: 35, 4: 10}, "down": {1: 120, 2: 120, 3: 83, 4: 43}},
        4: {"up": {1: 48, 2: 38, 3: 19, 4: 4}, "down": {1: 75, 2: 80, 3: 50, 4: 13}},
        5: {"up": {1: 93, 2: 63, 3: 35, 4: 10}, "down": {1: 120, 2: 120, 3: 83, 4: 43}},
        6: {"up": {1: 80, 2: 53, 3: 28, 4: 8}, "down": {1: 103, 2: 103, 3: 70, 4: 25}},
        7: {"up": {1: 80, 2: 53, 3: 28, 4: 8}, "down": {1: 103, 2: 103, 3: 70, 4: 25}},
        8: {"up": {1: 93, 2: 63, 3: 35, 4: 10}, "down": {1: 120, 2: 120, 3: 83, 4: 43}},
        9: {"up": {1: 80, 2: 53, 3: 28, 4: 8}, "down": {1: 103, 2: 103, 3: 70, 4: 25}},
        10: {"up": {1: 28, 2: 15, 3: 5, 4: 0}, "down": {1: 38, 2: 35, 3: 15, 4: 0}},
        11: {"up": {1: 45, 2: 28, 3: 12, 4: 2}, "down": {1: 60, 2: 60, 3: 31, 4: 3}},
        12: {"up": {1: 45, 2: 28, 3: 12, 4: 2}, "down": {1: 60, 2: 60, 3: 31, 4: 3}},
        13: {"up": {1: 28, 2: 15, 3: 5, 4: 0}, "down": {1: 38, 2: 35, 3: 15, 4: 0}},
    },
}

# Air Tankers Production Rates

AirTankerType = {
    "P-3": 1,
    "DC-7": 1,
    "C-130": 1,
    "DC-6": 2,
    "P2-V": 2,
    "S-2F": 3,
    "AT-802F": 3,
    "CL-215": 3,
    "CL-415": 3,
    "Air Tractor": 4,
    "Dromader": 4,
    "Thrush": 4,
}

# Rates = {Air Tanker Type: Min Capacity (gal)}
AirTankerRates = {1: 3000, 2: 2400, 3: 1300, 4: 800}

# Helicopter Types
HelicopterTypes = {
    "Bell-214": 1,
    "Bell-204": 2,
    "Bell-205": 2,
    "Bell-212": 2,
    "Bell-206": 3,
}

# Helicopter Production Rates
# Rates = {Helo Type: Retardant/Water carrying (gal)}
HelicopterRates = {1: 700, 2: 300, 3: 100}
