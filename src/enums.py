from pathlib import Path
from importlib import resources

from enum import auto, Enum, IntEnum

import numpy as np
from PIL import Image

with resources.path('assets.textures', 'terrain.jpg') as path:
    TERRAIN_TEXTURE_PATH: Path = path

DRY_TERRAIN_BROWN_IMG: Image.Image = Image.fromarray(
    np.full((10, 10, 3), (205, 133, 63), dtype=np.uint8))

BURNED_RGB_COLOR = (139, 69, 19)


class BurnStatus(IntEnum):
    UNBURNED = 0
    BURNING = auto()
    BURNED = auto()
    FIRELINE = auto()
    SCRATCHLINE = auto()
    WETLINE = auto()


class SpriteLayer(IntEnum):
    TERRAIN = 1
    FIRE = 2
    LINE = 3
    RESOURCE = 4


class GameStatus(Enum):
    QUIT = auto()
    RUNNING = auto()
