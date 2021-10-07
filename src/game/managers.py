from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import pygame
from pygame import sprite

from .sprites import Fire, Terrain
from ..enums import BurnStatus
from ..world.parameters import Environment, FuelParticle
from ..world.rothermel import compute_rate_of_spread


NewLocsType = Tuple[Tuple[int, int], Tuple[int, int],
                    Tuple[int, int], Tuple[int, int],
                    Tuple[int, int], Tuple[int, int],
                    Tuple[int, int], Tuple[int, int]]


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

    def _prune_sprites(self) -> None:
        '''
        Internal method to remove any Fire sprites whose durations have
        exceded the maximum allowed duration and mark self.fire_map as
        BURNED.
        '''
        # Use the expired sprites to mark self.fire_map as burned
        expired_sprites = list(
            filter(lambda x: x.duration>=self.max_fire_duration, self.sprites))
        for sprite in expired_sprites:
            x, y, _, _ = sprite.rect
            self.fire_map[y, x] = BurnStatus.BURNED

        # Remove the expired sprites
        self.sprites = list(
            filter(lambda x: x.duration<self.max_fire_duration, self.sprites))

    def _get_new_locs(self, x: int, y: int) -> NewLocsType:
        '''
        Get the 8-connected locations of the input (x,y) coordinate. This
        function will filter the points that are beyond the boundaries of
        the game screen and/or the points that are already BURNED or BURNING.

        Parameters:
            x: The x coordinate of the location
            y: The y coordinate of the location
        
        Returns:
            new_coords: A tuple of tuple of ints containing the adjacent
                        pixel locations that are UNBURNED and within the
                        scope of the game screen
        '''
        new_locs = ((x+1, y), (x+1, y+1), (x, y+1), (x-1, y+1),
                            (x-1, y), (x-1, y-1), (x, y-1), (x+1, y-1))
        # Make sure each new location/pixel is:
        #   UNBURNED
        #   Within the game screen boundaries
        filter_func = lambda p: \
            self.fire_map[p[1], p[0]]==BurnStatus.UNBURNED \
            and p[0] < self.fire_map.shape[1] and p[0] >=0 \
            and p[1] < self.fire_map.shape[0] and p[1] >=0
        new_locs = tuple(filter(filter_func, new_locs))
        return new_locs


class RothermelFireManager(FireManager):
    '''
    This FireManager will spread the fire based on the basic Rothermel
    model (https://www.fs.fed.us/rm/pubs_series/rmrs/gtr/rmrs_gtr371.pdf).
    '''
    def __init__(self, init_pos: Tuple[int, int], fire_size: int,
                 max_fire_duration: int, pixel_scale: int,
                 fuel_particle: FuelParticle, terrain: Terrain,
                 environment: Environment) -> None:
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
            pixel_scale: The amount of ft each pixel represents. This is needed
                         to track how much a fire has burned at a certain
                         location since it may take more than one update for
                         a pixel/location to catch on fire depending on the
                         rate of spread.
            fuel_particle: The parameters that describe the fuel particle
            terrain: The Terrain that describes the simulation/game
            environment: The Environment that describes the simulation/game
        
        Returns:
            None
        '''
        super().__init__(init_pos, fire_size, max_fire_duration)
        self.pixel_scale = pixel_scale
        self.fuel_particle = fuel_particle
        self.terrain = terrain
        self.environment = environment

        # Keep track of how much each pixel has burned.
        # This is needed since each pixel represents a specific number of feet
        # and it might take more than one update to burn
        self.burn_amounts = np.zeros_like(self.fire_map)

    def update(self) -> None:
        '''
        Update the spreading of the fires. This function will remove
        any fires that have exceded their duration and will spread fires
        that have reached the correct rate of spread for a long enough
        time.

        Arguments:
            None

        Returns:
            None
        '''
        # Remove all fires that are past the max duration
        self._prune_sprites()
        num_sprites = len(self.sprites)
        for sprite_idx in range(num_sprites):
            sprite = self.sprites[sprite_idx]
            x, y, _, _ = sprite.rect
            fuel_arr = self.terrain.fuel_arrs[y, x]
            new_locs = self._get_new_locs(x, y)

            for x_new, y_new in new_locs:
                loc = (x, y, fuel_arr.tile.z)
                fuel_arr_new = self.terrain.fuel_arrs[y, x]
                loc_new = (x_new, y_new, fuel_arr_new.tile.z)
                rate_of_spread = compute_rate_of_spread(loc, loc_new,
                                                        fuel_arr_new,
                                                        self.fuel_particle,
                                                        self.environment)
                self.burn_amounts[y, x] += rate_of_spread
                if self.burn_amounts[y,x] > self.pixel_scale:
                    new_sprite = Fire((x_new, y_new), self.fire_size)
                    self.sprites.append(new_sprite)
                    self.fire_map[y_new, x_new] = BurnStatus.BURNING


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
        self._prune_sprites()
        for sprite in self.sprites:
            x, y, w, h = sprite.rect
            # Create new sprites and mark the area as burning if
            # it has been burning long enough to spread
            if sprite.duration == self.rate_of_spread:
                # Fire can spread in 8 directions, need to check all of them
                # and verify they are unburned to start a new fire there
                new_locs = self._get_new_locs(x, y)
                for loc in new_locs:
                    new_sprite = Fire(loc, self.fire_size)
                    self.sprites.append(new_sprite)
                    self.fire_map[loc[1], loc[0]] = BurnStatus.BURNING
            else:
                continue