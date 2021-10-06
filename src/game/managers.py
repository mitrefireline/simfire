from typing import Tuple

import pygame

from .sprites import Fire


class FireManager():
    def __init__(self, init_pos: Tuple[int, int], fire_size: int) -> None:
        self.init_pos = init_pos
        self.fire_size = fire_size

        init_fire = Fire(self.init_pos, self.fire_size)
        self.sprites = pygame.sprite.RenderPlain([init_fire])