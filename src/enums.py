from enum import auto, IntEnum

import numpy as np
from PIL import Image

from .config import terrain_size


TERRAIN_TEXTURE_PATH: str = 'assets/textures/terrain.jpg'
FIRE_TEXTURE_PATH: str = 'assets/textures/flames.png'


DRY_TERRAIN_BROWN_IMG: Image.Image = Image.fromarray(
    np.full((terrain_size, terrain_size, 3), (205, 133, 63), dtype=np.uint8))


class BurnStatus(IntEnum):
    UNBURNED = auto()
    BURNING = auto()
    BURNED = auto()


class SpriteLayer(IntEnum):
    TERRAIN = 1
    FIRE = 2
    RESOURCE = 3