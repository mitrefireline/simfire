from enum import auto, Enum, IntEnum

import numpy as np
from PIL import Image

TERRAIN_TEXTURE_PATH: str = 'assets/textures/terrain.jpg'
FIRE_TEXTURE_PATH: str = 'assets/textures/flames.png'
FIRELINE_TEXTURE_PATH: str = 'assets/textures/fire_line.jpg'
SCRATCHLINE_TEXTURE_PATH: str = 'assets/textures/scratch_line.jpg'
WETLINE_TEXTURE_PATH: str = 'assets/textures/wet_line.jpg'

DRY_TERRAIN_BROWN_IMG: Image.Image = Image.fromarray(
    np.full((10, 10, 3), (205, 133, 63), dtype=np.uint8))

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
