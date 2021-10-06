from typing import Tuple

import numpy as np
import pygame

from .image import load_image, set_image_dryness
from .. import config as cfg
from ..enums import BurnStatus
from ..world.parameters import FuelArray


class GameTile(pygame.sprite.Sprite):
    def __init__(self, fuel_array: FuelArray, size: int,
                 pos: Tuple[int, int]) -> None:
        super().__init__()

        self.fuel_array = fuel_array
        self.size = size
        self.pos = pos

        self.image = load_image('assets/textures/terrain.jpg')
        self.image = pygame.transform.scale(self.image, (self.size, self.size))
        self.update_image_dryness()

        self.rect = self.image.get_rect()
        move_x = self.size * self.pos[1]
        move_y = self.size * self.pos[0]
        self.rect = self.rect.move(move_x, move_y)

        self.layer = 1
    
    def update(self, fire_map: np.ndarray) -> None:
        '''
        Check if fire is touching Tile and update what it looks like
        '''
        # Crop the fire_map to only the region of this tile
        x, y, w, h = self.rect
        fire_map = fire_map[y:y+h, x:x+w]
        # Update the pixels only if there is any burned area
        if np.any(fire_map == BurnStatus.BURNED):
            self.update_image_burned_area(fire_map)

    def update_image_burned_area(self, fire_map: np.ndarray) -> None:
        burn_map = fire_map == BurnStatus.BURNED
        arr = pygame.surfarray.pixels3d(self.image)
        arr[burn_map] = (139, 69, 19)

    def update_image_dryness(self) -> None:
        '''
        Determine the percent change to make the terrain look drier (i.e.
        more red/yellow/brown) by using the fuel array values. Then, update
        the image using the pygame surfarray functionality to modify the image
        in-place.

        Arguments:
            None

        Returns:
            None
        '''
        # Add the numbers after normalization
        # M_x is inverted because a lower value is more flammable
        color_change_pct = self.fuel_array.w_0 / cfg.w_0_max + \
                           self.fuel_array.delta / cfg.delta_max + \
                           (cfg.M_x_max - self.fuel_array.M_x) / cfg.M_x_max
        # Divide by 3 since there are 3 values
        color_change_pct /= 3

        arr = pygame.surfarray.pixels3d(self.image)
        arr[...,0] = np.clip((1+color_change_pct)*arr[...,0], 0, 255).astype(np.uint8)
        arr[...,0] = np.clip(0.75*(1+color_change_pct)*arr[...,0], 0, 255).astype(np.uint8)
        return

        set_image_dryness(self.image, color_change_pct)


class Fire(pygame.sprite.Sprite):
    def __init__(self, pos: Tuple[int, int], size: int) -> None:
        super().__init__()

        self.pos = pos
        self.size = size

        self.image = load_image('assets/textures/flames.png')
        self.image = pygame.transform.scale(self.image, (self.size, self.size))

        self.rect = self.image.get_rect()
        self.rect = self.rect.move(self.pos[1], self.pos[0])

        self.layer: int = 2

        # Record how many frames this sprite has been alive
        self.duration: int = 0


    def update(self) -> None:
        '''
        Check which squares the fire is on and adjacent to and update its
        spread.

        Arguments:
            None

        Returns:
            None 
        '''
        # Increment the duration
        self.duration += 1