from dataclasses import astuple
from typing import List, Tuple

import numpy as np

from ..sprites import Fire, Terrain
from ...enums import BurnStatus, GameStatus
from ...world.parameters import Environment, FuelParticle
from ...world.rothermel import compute_rate_of_spread

NewLocsType = Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int],
                    Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]


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

    def update(self, fire_map: np.ndarray) -> None:
        '''
        Method that describes how the fires in self.sprites should spread.
        '''
        pass

    def _prune_sprites(self, fire_map: np.ndarray) -> np.ndarray:
        '''
        Internal method to remove any Fire sprites whose durations have
        exceded the maximum allowed duration and mark self.fire_map as
        BURNED.
        '''
        # Use the expired sprites to mark self.fire_map as burned
        expired_sprites = list(
            filter(lambda x: x.duration >= self.max_fire_duration, self.sprites))
        for sprite in expired_sprites:
            x, y, _, _ = sprite.rect
            fire_map[y, x] = BurnStatus.BURNED

        # Remove the expired sprites
        self.sprites = list(
            filter(lambda x: x.duration < self.max_fire_duration, self.sprites))

        return fire_map

    def _get_new_locs(self, x: int, y: int, fire_map: np.ndarray) -> NewLocsType:
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
        def _filter_function(loc: Tuple[int, int]) -> bool:
            '''Used in `self.get_new_locs` as the filter function for the new locations

            Make sure each new location/pixel is:
            - Within the game screen boundaries
            - UNBURNED
            '''
            in_boundaries = loc[0] < fire_map.shape[1] \
                            and loc[0] >= 0 \
                            and loc[1] < fire_map.shape[0] \
                            and loc[1] >= 0 \
                            and fire_map[loc[1], loc[0]] == BurnStatus.UNBURNED
            return in_boundaries

        new_locs = ((x + 1, y), (x + 1, y + 1), (x, y + 1), (x - 1, y + 1), (x - 1, y),
                    (x - 1, y - 1), (x, y - 1), (x + 1, y - 1))
        # Make sure each new location/pixel is:
        #   Within the game screen boundaries
        #   UNBURNED
        new_locs = tuple(filter(_filter_function, new_locs))

        return new_locs


class RothermelFireManager(FireManager):
    '''
    This FireManager will spread the fire based on the basic Rothermel
    model (https://www.fs.fed.us/rm/pubs_series/rmrs/gtr/rmrs_gtr371.pdf).
    '''
    def __init__(self, init_pos: Tuple[int, int], fire_size: int, max_fire_duration: int,
                 pixel_scale: int, fuel_particle: FuelParticle, terrain: Terrain,
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
        self.burn_amounts = np.zeros_like(self.terrain.fuel_arrs)
        self.slope_mag, self.slope_dir = self._compute_slopes()

    def _compute_slopes(self) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Compute the gradient/slope magnitude and direction for every point for use
        with the Rothermel calculation. This is easier than computing on the fly for
        every new location in the update() call.

        Arguments:
            None

        Returns:
            grad_mag: The gradient/slope magnitude for every pixel
            grad_dir: The gradient direction/angle for every pixel
        '''
        grad_y, grad_x = np.gradient(self.terrain.elevations)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_dir = np.tan(grad_y / (grad_x + 0.000001))
        return grad_mag, grad_dir

    def update(self, fire_map: np.ndarray) -> Tuple[np.ndarray, GameStatus]:
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
        self._prune_sprites(fire_map)
        num_sprites = len(self.sprites)
        if num_sprites == 0:
            return GameStatus.QUIT
        loc_x = []
        loc_y = []
        new_loc_x = []
        new_loc_y = []
        w_0 = []
        delta = []
        M_x = []
        sigma = []
        h = []
        S_T = []
        S_e = []
        p_p = []
        M_f = []
        U = []
        U_dir = []
        slope_mag = []
        slope_dir = []
        for sprite_idx in range(num_sprites):
            sprite = self.sprites[sprite_idx]
            x, y, _, _ = sprite.rect
            new_locs = self._get_new_locs(x, y, fire_map)
            num_locs = len(new_locs)
            if num_locs == 0:
                continue

            new_locs_uzip = tuple(zip(*new_locs))
            new_loc_x.extend(new_locs_uzip[0])
            new_loc_y.extend(new_locs_uzip[1])
            loc_x.extend([x] * num_locs)
            loc_y.extend([y] * num_locs)

            n_w_0, n_delta, n_M_x, n_sigma = list(
                zip(*[
                    astuple(arr.fuel)
                    for arr in self.terrain.fuel_arrs[new_locs_uzip[::-1]]
                ]))
            w_0.extend(n_w_0)
            delta.extend(n_delta)
            M_x.extend(n_M_x)
            sigma.extend(n_sigma)

            # Set the FuelParticle parameters into arrays
            h.extend([self.fuel_particle.h] * num_locs)
            S_T.extend([self.fuel_particle.S_T] * num_locs)
            S_e.extend([self.fuel_particle.S_e] * num_locs)
            p_p.extend([self.fuel_particle.p_p] * num_locs)

            # Set the Environment parameters into arrays
            M_f.extend([self.environment.M_f] * num_locs)
            U.extend([self.environment.U] * num_locs)
            U_dir.extend([self.environment.U_dir] * num_locs)

            # Set the slope parameters into arrays
            slope_mag.extend(self.slope_mag[new_locs_uzip[::-1]].tolist())
            slope_dir.extend(self.slope_dir[new_locs_uzip[::-1]].tolist())

        loc_x = np.array(loc_x, dtype=np.float32)
        loc_y = np.array(loc_y, dtype=np.float32)
        new_loc_x = np.array(new_loc_x, dtype=np.float32)
        new_loc_y = np.array(new_loc_y, dtype=np.float32)
        w_0 = np.array(w_0, dtype=np.float32)
        delta = np.array(delta, dtype=np.float32)
        M_x = np.array(M_x, dtype=np.float32)
        sigma = np.array(sigma, dtype=np.float32)
        h = np.array(h, dtype=np.float32)
        S_T = np.array(S_T, dtype=np.float32)
        S_e = np.array(S_e, dtype=np.float32)
        p_p = np.array(p_p, dtype=np.float32)
        M_f = np.array(M_f, dtype=np.float32)
        U = np.array(U, dtype=np.float32)
        U_dir = np.array(U_dir, dtype=np.float32)
        slope_mag = np.array(slope_mag, dtype=np.float32)
        slope_dir = np.array(slope_dir, dtype=np.float32)

        R = compute_rate_of_spread(loc_x, loc_y, new_loc_x, new_loc_y, w_0, delta, M_x,
                                   sigma, h, S_T, S_e, p_p, M_f, U, U_dir, slope_mag,
                                   slope_dir)

        y_coords = new_loc_y.astype(int)
        x_coords = new_loc_x.astype(int)
        self.burn_amounts[y_coords, x_coords] += R

        y_coords, x_coords = np.unique(np.vstack((y_coords, x_coords)), axis=1)
        for (x_new, y_new) in zip(x_coords, y_coords):
            if self.burn_amounts[y_new, x_new] > self.pixel_scale:
                new_sprite = Fire((x_new, y_new), self.fire_size)
                self.sprites.append(new_sprite)
                fire_map[y_new, x_new] = BurnStatus.BURNING

        return fire_map, GameStatus.RUNNING


class ConstantSpreadFireManager(FireManager):
    '''
    This FireManager will spread fire at a constant rate in all directions.
    '''
    def __init__(self, init_pos: Tuple[int, int], fire_size: int, max_fire_duration: int,
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

    def update(self, fire_map: np.ndarray) -> None:
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
        self._prune_sprites(fire_map)
        for sprite in self.sprites:
            x, y, _, _ = sprite.rect
            # Create new sprites and mark the area as burning if
            # it has been burning long enough to spread
            if sprite.duration == self.rate_of_spread:
                # Fire can spread in 8 directions, need to check all of them
                # and verify they are unburned to start a new fire there
                new_locs = self._get_new_locs(x, y, fire_map)
                for loc in new_locs:
                    new_sprite = Fire(loc, self.fire_size)
                    self.sprites.append(new_sprite)
                    fire_map[loc[1], loc[0]] = BurnStatus.BURNING
            else:
                continue

        return fire_map
