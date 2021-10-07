from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import pygame

from .sprites import Fire
from ..enums import BurnStatus


class FireManager():
    '''
    Base class to manage the spread of the fire as well as the map of which
    pixels have already burned. Child classes should create their own update()
    method to describe how the fire spreads.
    '''
    def __init__(self, init_pos: Tuple[int, int], fire_size: int,
                 max_fire_duration: int) -> None:
        '''
        Initialize the class by recording the initial fire location and size.
        Create the fire sprite and fire_map and mark the location of the
        initial fire. 

        Arguments:
            init_pos: The (x,y) location of the initial fire
            fire_size: The (n,n) pixel size of the fire sprite. Note that
                       the sprite pixel size does not affect which tiles/pixels
                       are actually burning. This is for display purposes only.
                       Each fire is only burning on one pixel at a time.
            max_fire_duration: The number of frames/updates a fire will burn
                               for before going out. This is moslty useful so
                               that fires that have spread and are now on the
                               interior do not have to keep being rendered.
        
        Returns:
            None
        '''
        self.init_pos = init_pos
        self.fire_size = fire_size
        self.max_fire_duration = max_fire_duration

        init_fire = Fire(self.init_pos, self.fire_size)
        self.sprites: List[Fire] = [init_fire]

        # Map to track which pixels are on fire or have burned
        self.fire_map = np.full(pygame.display.get_surface().get_size(),
                                BurnStatus.UNBURNED)
        self.fire_map[init_pos[1], init_pos[0]] = BurnStatus.BURNING

    def update(self) -> None:
        '''
        Method that describes how the fires in self.sprites should spread. 
        '''
        pass


class ConstantSpreadFireManager(FireManager):
    '''
    This FireManager will spread fire at a constant rate in all directions. 
    '''
    def __init__(self, init_pos: Tuple[int, int], fire_size: int,
                 max_fire_duration: int,
                 rate_of_spread: int) -> None:
        '''
        Initialize the class by recording the initial fire location and size.
        Create the fire sprite and fire_map and mark the location of the
        initial fire. 

        Arguments:
            init_pos: The (x,y) location of the initial fire
            fire_size: The (n,n) pixel size of the fire sprite. Note that
                       the sprite pixel size does not affect which tiles/pixels
                       are actually burning. This is for display purposes only.
                       Each fire is only burning on one pixel at a time.
            max_fire_duration: The number of frames/updates a fire will burn
                               for before going out. This is moslty useful so
                               that fires that have spread and are now on the
                               interior do not have to keep being rendered.
            rate_of_spread: The number of frames that must pass before a fire
                            can spread to adjacent pixels.
        
        Returns:
            None
        '''
        super().__init__(init_pos, fire_size, max_fire_duration)
        self.rate_of_spread = rate_of_spread

    def update(self) -> None:
        '''
        Spread the fire and add new fire sprites to the display. The fire
        will spread at a constant rate in all directions based on
        self.rate_of_spread. self.fire_map is updated accordingly based on
        which fires have started/spread/stopped.

        Arguments:
            None

        Returns:
            None
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
                    if (loc[0] < 0) or (loc[0] >= self.fire_map.shape[1]):
                        continue
                    if (loc[1] < 0) or (loc[1] >= self.fire_map.shape[0]):
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