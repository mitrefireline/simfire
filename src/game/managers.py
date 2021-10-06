from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pygame

from .sprites import Fire
from .. import config as cfg
from ..enums import BurnStatus



class FireManager():
    def __init__(self, init_pos: Tuple[int, int], fire_size: int,
                 max_fire_duration: int,
                 rate_of_spread: int) -> None:
        self.init_pos = init_pos
        self.fire_size = fire_size
        self.max_fire_duration = max_fire_duration
        self.rate_of_spread = rate_of_spread

        init_fire = Fire(self.init_pos, self.fire_size)
        self.sprites: List[Fire] = [init_fire]

        # Map to track which pixels are on fire or have burned
        self.fire_map = np.full(pygame.display.get_surface().get_size(),
                                BurnStatus.UNBURNED)
        self.fire_map[init_pos[1], init_pos[0]] = BurnStatus.BURNING

    def update(self) -> None:
        '''
        Spread the fire and add new fire sprites to the display 
        '''
        for sprite in self.sprites:
            x, y, w, h = sprite.rect
            # Remove the fire sprite and mark the area as burned if
            # it has been burning for longer than the max duration
            if sprite.duration > self.max_fire_duration:
                self.sprites.remove(sprite)
                self.fire_map[y, x] = BurnStatus.BURNED
            # Create new sprites mark the area as burning if
            # it has been burning long enough to spread
            elif sprite.duration == self.rate_of_spread:
                # Fire can spread in 8 directions, need to check all of them
                # and verify they are unburned to start a new fire there
                new_locs = ((x+1, y), (x+1, y+1), (x, y+1), (x-1, y+1),
                            (x-1, y), (x-1, y-1), (x, y-1), (x+1, y-1))
                for loc in new_locs:
                    # Skip the location if it is past the boundary of the map
                    if loc[0] >= self.fire_map.shape[1]:
                        continue
                    if loc[1] >= self.fire_map.shape[0]:
                        continue
                    # Spawn a new sprite and mark the area
                    # as burning if it is currently unburned
                    if self.fire_map[loc[1], loc[0]] == BurnStatus.UNBURNED:
                        new_sprite = Fire(loc, self.fire_size)
                        self.sprites.append(new_sprite)
                        self.fire_map[loc[1], loc[0]] = BurnStatus.BURNING
                    else:
                        continue
            else:
                continue