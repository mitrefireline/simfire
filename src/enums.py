from enum import auto, Enum, IntEnum

import numpy as np
from PIL import Image

from .config import terrain_size

TERRAIN_TEXTURE_PATH: str = 'assets/textures/terrain.jpg'
FIRE_TEXTURE_PATH: str = 'assets/textures/flames.png'
FIRELINE_TEXTURE_PATH: str = 'assets/textures/fire_line.jpg'

DRY_TERRAIN_BROWN_IMG: Image.Image = Image.fromarray(
    np.full((terrain_size, terrain_size, 3), (205, 133, 63), dtype=np.uint8))

BURNED_RGB_COLOR = (139, 69, 19)


class BurnStatus(IntEnum):
    UNBURNED = auto()
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
