'''
Fire
====

Defines the different `FireManager`s (`ConstantSpreadFireManager` and
`RothermelFireManager`) that determine how a fire moves about a `fire_map`.
'''
from dataclasses import astuple
from typing import List, Tuple, Union

import numpy as np

from ..sprites import Fire, Terrain
from ...enums import BurnStatus, RoSAttenuation, GameStatus
from ...world.parameters import Environment, FuelParticle
from ...world.rothermel import compute_rate_of_spread

NewLocsType = Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int],
                    Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]

SpriteParamsType = List[Union[List[float], Tuple[float]]]


class FireManager():
    '''
    Base class to manage the spread of the fire as well as the map of which
    pixels have already burned. Child classes should create their own update()
    method to describe how the fire spreads.
    '''
    def __init__(self,
                 init_pos: Tuple[int, int],
                 fire_size: int,
                 max_fire_duration: int,
                 attenuate_line_ros: bool = True) -> None:
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
            attenuate_line_ros: Whether or not to attenuate the rate of spread.
                                Defaults to `True`. If set to `True`, will subtract
                                values found in `enums.RoSAttenuation` from the initial
                                rate of spread calculation. If set to `False`, all
                                different control lines will completely stop the fire.
        Returns:
            None
        '''
        self.init_pos = init_pos
        self.fire_size = fire_size
        self.max_fire_duration = max_fire_duration
        self.attenuate_line_ros = attenuate_line_ros

        init_fire = Fire(self.init_pos, self.fire_size)
        self.sprites: List[Fire] = [init_fire]
        self.durations: List[int] = [0]

    def update(self, fire_map: np.ndarray) -> None:
        '''
        Method that describes how the fires in `self.sprites`should spread.

        Should be updated in child classes.

        Arguments:
            fire_map: All possible fire conditions at each pixel location
        '''
        pass

    def _prune_sprites(self, fire_map: np.ndarray) -> np.ndarray:
        '''
        Internal method to remove any Fire sprites whose durations have
        exceded the maximum allowed duration and mark `self.fire_map` as
        BURNED.

        Arguments:
            fire_map: All possible fire conditions at each pixel location

        Returns:
            An updated `fire_map` with sprites pruned
        '''
        lists_zipped = list(zip(self.sprites, self.durations))
        # Get the sprites whose duration exceeds the max allowed duration
        results = list(filter(lambda x: x[1] >= self.max_fire_duration, lists_zipped))
        results = list(zip(*results))
        if len(results) > 0:
            expired_sprites = results[0]
            # Use the expired sprites to mark self.fire_map as burned
            for sprite in expired_sprites:
                x, y, _, _ = sprite.rect
                fire_map[y, x] = BurnStatus.BURNED

        # Remove the expired sprites
        results = list(filter(lambda x: x[1] < self.max_fire_duration, lists_zipped))

        if len(results) > 0:
            self.sprites, self.durations = list(map(list, zip(*results)))
        else:
            self.sprites = []
            self.durations = []

        return fire_map

    def _get_new_locs(self, x: int, y: int, fire_map: np.ndarray) -> NewLocsType:
        '''
        Get the 8-connected locations of the input (x,y) coordinate. This
        function will filter the points that are beyond the boundaries of
        the game screen and/or the points that are already `BURNED` or `BURNING`.

        Arguments:
            x: The x coordinate of the location
            y: The y coordinate of the location

        Returns:
            A tuple of tuple of ints containing the adjacent
            pixel locations that are `UNBURNED` and within the
            scope of the game screen
        '''
        def _filter_function(loc: Tuple[int, int]) -> bool:
            '''Used in `self.get_new_locs` as the filter function for the new locations

            Make sure each new location/pixel is:
                - Within the game screen boundaries
                - UNBURNED

            Arguments:
                loc: The tuple of the x and y location to filter

            Returns:
                Whether or not the `loc` was inside the boundaries
            '''
            acceptable_statuses = [
                BurnStatus.UNBURNED, BurnStatus.FIRELINE, BurnStatus.SCRATCHLINE,
                BurnStatus.WETLINE
            ]

            in_boundaries = loc[0] < fire_map.shape[1] \
                            and loc[0] >= 0 \
                            and loc[1] < fire_map.shape[0] \
                            and loc[1] >= 0 \
                            and fire_map[loc[1], loc[0]] in acceptable_statuses
            return in_boundaries

        new_locs = ((x + 1, y), (x + 1, y + 1), (x, y + 1), (x - 1, y + 1), (x - 1, y),
                    (x - 1, y - 1), (x, y - 1), (x + 1, y - 1))
        # Make sure each new location/pixel is:
        #   Within the game screen boundaries
        #   UNBURNED
        new_locs = tuple(filter(_filter_function, new_locs))

        return new_locs

    def _update_rate_of_spread(self, rate_of_spread: np.ndarray,
                               fire_map: np.ndarray) -> np.ndarray:
        '''Update the burn amounts based on control line status.

        This will subtract the rate of spread for all control line locations in
        `rate_of_spread` by the numbers set in `enums.RosAttenuation`.

        e.g. if the `rate_of_spread` (RoS) was 10 for a location where a fireline was
        located, and `RosAttenuation.FIRELINE` was set to 6, it would become 4 as a
        result of this function.

        Arguments:
            rate_of_spread: The array that keeps track of the rate of spread for
                          for all pixel locations.
            fire_map: The array that maintains information about the status of the fire
                      at each pixel location (`BURNED`, `UNBURNED`, `FIRELINE`, etc.)

        Returns:
            An updated `rate_of_spread` array, taking into account the control lines
        '''
        assert fire_map.shape == rate_of_spread.shape

        factor = np.zeros_like(rate_of_spread)
        if self.attenuate_line_ros:
            factor[np.where(fire_map == BurnStatus.FIRELINE)] = RoSAttenuation.FIRELINE
            factor[np.where(
                fire_map == BurnStatus.SCRATCHLINE)] = RoSAttenuation.SCRATCHLINE
            factor[np.where(fire_map == BurnStatus.WETLINE)] = RoSAttenuation.WETLINE
        else:
            factor[np.where(fire_map == BurnStatus.FIRELINE)] = 0
            factor[np.where(fire_map == BurnStatus.SCRATCHLINE)] = 0
            factor[np.where(fire_map == BurnStatus.WETLINE)] = 0

        rate_of_spread = rate_of_spread - factor

        return rate_of_spread


class RothermelFireManager(FireManager):
    '''
    This FireManager will spread the fire based on the basic [Rothermel
    Model](https://www.fs.fed.us/rm/pubs_series/rmrs/gtr/rmrs_gtr371.pdf).
    '''
    def __init__(self,
                 init_pos: Tuple[int, int],
                 fire_size: int,
                 max_fire_duration: int,
                 pixel_scale: int,
                 update_rate: float,
                 fuel_particle: FuelParticle,
                 terrain: Terrain,
                 environment: Environment,
                 attenuate_line_ros: bool = True) -> None:
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
            update_rate: The amount of time in minutes that passes for each simulation
                         update step
            fuel_particle: The parameters that describe the fuel particle
            terrain: The Terrain that describes the simulation/game
            environment: The Environment that describes the simulation/game
            attenuate_line_ros: Whether or not to attenuate the rate of spread.
                                Defaults to `True`. If set to `True`, will subtract
                                values found in `enums.RoSAttenuation` from the initial
                                rate of spread calculation. If set to `False`, all
                                different control lines will completely stop the fire.

        Returns:
            None
        '''
        super().__init__(init_pos, fire_size, max_fire_duration, attenuate_line_ros)
        self.pixel_scale = pixel_scale
        self.update_rate = update_rate
        self.fuel_particle = fuel_particle
        self.terrain = terrain

        self.environment = environment
        self.environment.U = np.asarray(environment.U, dtype=np.float32)
        self.environment.U_dir = np.asarray(environment.U, dtype=np.float32)

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
            The gradient/slope magnitude for every pixel ([0]) and the gradient
            direction/angle for every pixel ([1])
        '''
        grad_y, grad_x = np.gradient(self.terrain.elevations)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_dir = np.tan(grad_y / (grad_x + 0.000001))
        return grad_mag, grad_dir

    def _accrue_sprites(self, sprite_idx: int, fire_map: np.ndarray) -> SpriteParamsType:
        '''
        Pull all neccessary information for the update step in a multiprocessable way.
        This will return a list of lists containing the Rothermel computation information
        for a single sprite and all of its possible new locations.

        Arguments:
            sprite_idx: position of the sprite
            fire_map: The numpy array that tracks the fire's burn status for
                      each pixel in the simulation

        Returns:
            The sprite parameters needed for a Rothermel calculation for each sprite for
            each possible new spreadable fire location
        '''

        sprite = self.sprites[sprite_idx]
        x, y, _, _ = sprite.rect
        new_locs = self._get_new_locs(x, y, fire_map)
        num_locs = len(new_locs)
        if num_locs == 0:
            return

        new_locs_uzip = tuple(zip(*new_locs))
        new_loc_x = new_locs_uzip[0]
        new_loc_y = new_locs_uzip[1]
        loc_x = [x] * num_locs
        loc_y = [y] * num_locs
        n_w_0, n_delta, n_M_x, n_sigma = list(
            zip(*[
                astuple(arr.fuel) for arr in self.terrain.fuel_arrs[new_locs_uzip[::-1]]
            ]))
        # Set the FuelParticle parameters into arrays
        h = [self.fuel_particle.h] * num_locs
        S_T = [self.fuel_particle.S_T] * num_locs
        S_e = [self.fuel_particle.S_e] * num_locs
        p_p = [self.fuel_particle.p_p] * num_locs
        # Set the Environment parameters into arrays
        M_f = [self.environment.M_f] * num_locs
        U = []
        U.extend(list(self.environment.U[new_locs_uzip[::-1]]))
        U_dir = []
        U_dir.extend(list(self.environment.U_dir[new_locs_uzip[::-1]]))
        # Set the slope parameters into arrays
        slope_mag = self.slope_mag[new_locs_uzip[::-1]].tolist()
        slope_dir = self.slope_dir[new_locs_uzip[::-1]].tolist()

        return [
            loc_x, loc_y, new_loc_x, new_loc_y, n_w_0, n_delta, n_M_x, n_sigma, h, S_T,
            S_e, p_p, M_f, U, U_dir, slope_mag, slope_dir
        ]

    def _flatten_params(self, all_params: SpriteParamsType) -> List[np.ndarray]:
        '''
        Flatten the sprite parameters into an array of shape
        (num_parameters, num_points_to_compute). This will allow for the Rothermel
        calculation to be done in a multiprocessable way.

        Arguments:
            all_params: The sprite parameters needed for a Rothermel calculation for
                        each sprite for each possible new spreadable fire location. This
                        will typically be computed by self._accrue_sprites for each
                        sprite

        Returns:
            The sprtie parameters with shape (num_parameters, num_points_to_compute).
            The input is transformed from a list of lists/tuples into a 2D array
            containing the information in a vectorized/multiprocessing format
        '''
        if len(self.sprites) == 1:  # single burning pixel case (first sim step typically)
            arr = np.asarray(all_params, dtype=np.float32)
            arr = np.reshape(arr, (arr.shape[1], arr.shape[0] * arr.shape[2]))
        else:  # Multiple burning pixels
            num_params_per_example = len(all_params[0])
            arr = [
                np.hstack([x[i] for x in all_params])
                for i in range(num_params_per_example)
            ]
            arr = np.asarray(arr, dtype=np.float32)

        return [arr[i, :] for i in range(arr.shape[0])]

    def update(self, fire_map: np.ndarray) -> Tuple[np.ndarray, GameStatus]:
        '''
        Update the spreading of the fires. This function will remove
        any fires that have exceded their duration and will spread fires
        that have reached the correct rate of spread for a long enough
        time.

        Arguments:
            fire_map: The numpy array that tracks the fire's burn status for
                      each pixel in the simulation

        Returns:
            A NumPy array of the updated `fire_map` and the current `GameStatus`
        '''
        # Remove all fires that are past the max duration
        self._prune_sprites(fire_map)
        # Increment the durations
        self.durations = list(map(lambda x: x + 1, self.durations))
        num_sprites = len(self.sprites)
        if num_sprites == 0:
            return fire_map, GameStatus.QUIT
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
        sprite_idxs = list(range(num_sprites))

        all_params = [self._accrue_sprites(idx, fire_map) for idx in sprite_idxs]
        all_params = list(filter(None, all_params))

        # Sprites exist, but there are no new locations to spread to
        if len(all_params) == 0:
            return fire_map, GameStatus.RUNNING

        [
            loc_x, loc_y, new_loc_x, new_loc_y, w_0, delta, M_x, sigma, h, S_T, S_e, p_p,
            M_f, U, U_dir, slope_mag, slope_dir
        ] = self._flatten_params(all_params)

        R = compute_rate_of_spread(loc_x, loc_y, new_loc_x, new_loc_y, w_0, delta, M_x,
                                   sigma, h, S_T, S_e, p_p, M_f, U, U_dir, slope_mag,
                                   slope_dir)

        # Scale the rate of spread by the update rate
        R *= self.update_rate

        y_coords = new_loc_y.astype(int)
        x_coords = new_loc_x.astype(int)

        rate_of_spread = np.zeros_like(self.burn_amounts)
        rate_of_spread[y_coords, x_coords] = R

        # Update the burn_amounts dependent on if there are control lines there
        # And only update if specified in the class
        rate_of_spread = self._update_rate_of_spread(rate_of_spread, fire_map)
        self.burn_amounts += rate_of_spread

        y_coords, x_coords = np.unique(np.vstack((y_coords, x_coords)), axis=1)
        new_burn = np.argwhere(self.burn_amounts[y_coords, x_coords] > self.pixel_scale)
        new_sprites = [
            Fire((x_coords[burn[0]], y_coords[burn[0]]), self.fire_size)
            for burn in new_burn
        ]
        new_durations = [0] * len(new_sprites)

        self.sprites = self.sprites + new_sprites
        self.durations = self.durations + new_durations
        fire_map[y_coords[new_burn], x_coords[new_burn]] = BurnStatus.BURNING

        return fire_map, GameStatus.RUNNING


class ConstantSpreadFireManager(FireManager):
    '''
    This FireManager will spread fire at a constant rate in all directions.
    '''
    def __init__(self, init_pos: Tuple[int, int], fire_size: int, max_fire_duration: int,
                 rate_of_spread: int) -> None:
        '''
        Initialize the class by recording the initial fire location and size.
        Create the fire sprite and `fire_map` and mark the location of the
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

    def update(self, fire_map: np.ndarray) -> np.ndarray:
        '''
        Spread the fire and add new fire sprites to the display. The fire
        will spread at a constant rate in all directions based on
        self.rate_of_spread. `self.fire_map` is updated accordingly based on
        which fires have started/spread/stopped.

        Arguments:
            fire_map: All possible fire conditions at each pixel location

        Returns:
            An updated NumPy array of the current `fire_map`
        '''
        self._prune_sprites(fire_map)
        for sprite, duration in zip(self.sprites, self.durations):
            x, y, _, _ = sprite.rect
            # Create new sprites and mark the area as burning if
            # it has been burning long enough to spread
            if duration == self.rate_of_spread:
                # Fire can spread in 8 directions, need to check all of them
                # and verify they are unburned to start a new fire there
                new_locs = self._get_new_locs(x, y, fire_map)
                for loc in new_locs:
                    new_sprite = Fire(loc, self.fire_size)
                    self.sprites.append(new_sprite)
                    fire_map[loc[1], loc[0]] = BurnStatus.BURNING
            else:
                continue

        # Increment the durations
        self.durations = list(map(lambda x: x + 1, self.durations))

        return fire_map
