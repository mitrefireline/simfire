import numpy as np

import unittest
from typing import Dict
from pathlib import Path

from ...game.sprites import Terrain
from ...utils.config import Config
from ...world.parameters import Environment
from ...rl_env.simulation import RothermelSimulation
from ...enums import BurnStatus
from ...game.managers.mitigation import (FireLineManager, ScratchLineManager,
                                         WetLineManager)
from ...game.managers.fire import RothermelFireManager


class RothermelSimulationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config('./src/utils/_tests/test_configs/test_config.yml')
        self.config_flat_simple = Config(
            Path(self.config.path).parent / 'test_config_flat_simple.yml')

        self.screen_size = (self.config.area.screen_size, self.config.area.screen_size)

        self.simulation = RothermelSimulation(self.config)
        self.simulation_flat = RothermelSimulation(self.config_flat_simple)

        topo_layer = self.config.terrain.elevation_function
        fuel_layer = self.config.terrain.fuel_array_function
        self.terrain = Terrain(fuel_layer, topo_layer, self.screen_size)
        self.simulation.terrain = self.terrain
        self.simulation.environment = Environment(self.config.environment.moisture,
                                                  self.config.wind.speed,
                                                  self.config.wind.direction)

        # initialize all mitigation strategies
        self.simulation.fireline_manager = FireLineManager(
            size=self.config.display.control_line_size,
            pixel_scale=self.config.area.pixel_scale,
            terrain=self.simulation.terrain,
            headless=self.config.simulation.headless)

        self.simulation.scratchline_manager = ScratchLineManager(
            size=self.config.display.control_line_size,
            pixel_scale=self.config.area.pixel_scale,
            terrain=self.simulation.terrain,
            headless=self.config.simulation.headless)

        self.simulation.wetline_manager = WetLineManager(
            size=self.config.display.control_line_size,
            pixel_scale=self.config.area.pixel_scale,
            terrain=self.simulation.terrain,
            headless=self.config.simulation.headless)

        self.simulation.fireline_sprites = self.simulation.fireline_manager.sprites
        self.simulation.fireline_sprites_empty = self.simulation.fireline_sprites.copy()
        self.simulation.scratchline_sprites = self.simulation.scratchline_manager.sprites
        self.simulation.wetline_sprites = self.simulation.wetline_manager.sprites

        self.simulation.fire_manager = RothermelFireManager(
            self.config.fire.fire_initial_position,
            self.config.display.fire_size,
            self.config.fire.max_fire_duration,
            self.config.area.pixel_scale,
            self.config.simulation.update_rate,
            self.simulation.fuel_particle,
            self.simulation.terrain,
            self.simulation.environment,
            max_time=self.config.simulation.runtime,
            attenuate_line_ros=self.config.mitigation.ros_attenuation,
            headless=self.config.simulation.headless)
        self.simulation.fire_sprites = self.simulation.fire_manager.sprites

    def test__create_terrain(self) -> None:
        '''
        Test that the terrain gets created properly.

        This should work as long as the tests for Terrain() work.
        '''
        pass

    def test__create_mitigations(self) -> None:
        '''
        Test that the mitigation (FireLineManager) gets created properly.

        This should work as long as the tests for FireLineManager() work.
        '''
        pass

    def test_create_fire(self) -> None:
        '''
        Test that the fire (FireManager) gets created properly.

        This should work as long as the tests for FireManager() work.
        '''
        pass

    def test_get_actions(self) -> None:
        '''
        Test that the call to `get_actions()` runs properly and returns all Rothermel
        `FireLineManager()` features.
        '''
        simulation_actions = self.simulation.get_actions()
        self.assertIsInstance(simulation_actions, Dict)

    def test_get_attributes(self) -> None:
        '''
        Test that the call to get_actions() runs properly and returns all Rothermel
        features (Fire, Wind, FireLine, Terrain).
        '''
        simulation_attributes = self.simulation.get_attributes()
        self.assertIsInstance(simulation_attributes, Dict)

    def test__update_sprite_points(self) -> None:
        '''
        Test that the call to `_update_sprites()` runs through properly.

        Since self.action is instantiated as `1`, we need to verify that a fireline sprite
        is created and added to the `fireline_manager`.
        '''

        # assert points get updated 'inline' as agent traverses
        self.config.render.inline = True
        current_agent_loc = ([1], [1])
        self.mitigation = ([BurnStatus.FIRELINE])
        points = {(1, 1)}
        self.simulation._update_sprite_points(self.mitigation, current_agent_loc)
        self.assertEqual(self.simulation.points,
                         points,
                         msg=f'The sprite was updated at {self.simulation.points}, '
                         f'but it should have been at {current_agent_loc}')

        # do the previous one again but with a SCRATCHLINE
        self.mitigation = ([BurnStatus.SCRATCHLINE])
        self.simulation._update_sprite_points(self.mitigation, current_agent_loc)
        self.assertEqual(self.simulation.points,
                         points,
                         msg=f'The sprite was updated at {self.simulation.points}, '
                         f'but it should have been at {current_agent_loc}')

        # do the previous one again but with a WETLINE
        self.mitigation = ([BurnStatus.WETLINE])
        self.simulation._update_sprite_points(self.mitigation, current_agent_loc)
        self.assertEqual(self.simulation.points,
                         points,
                         msg=f'The sprite was updated at {self.simulation.points}, '
                         f'but it should have been at {current_agent_loc}')

        # assert points get updated after agent traverses entire game
        self.config.render.inline = False
        self.config.render.post_agent = True
        current_agent_loc = (self.config.area.screen_size, self.config.area.screen_size)
        self.mitigation = np.full(
            (self.config.area.screen_size, self.config.area.screen_size),
            BurnStatus.FIRELINE)
        mitigations = np.array(
            [BurnStatus.FIRELINE, BurnStatus.SCRATCHLINE, BurnStatus.WETLINE])
        self.mitigation = mitigations[np.random.randint(
            len(mitigations),
            size=(self.config.area.screen_size, self.config.area.screen_size))]
        points = [(i, j) for j in range(self.config.area.screen_size)
                  for i in range(self.config.area.screen_size)]
        points = set(points)
        self.simulation._update_sprite_points(self.mitigation)
        self.assertEqual(
            self.simulation.points,
            points,
            msg=f'The number of sprites updated was {len(self.simulation.points)} '
            f', but it should have been {len(points)} sprites.')

    def test_run(self) -> None:
        '''
        Test that the call to `_run` runs the simulation properly.

        This function returns the burned firemap with or w/o mitigation.

        This function will reset the `fire_map` to all `UNBURNED` pixels at each call to
        the method.

        This should pass as long as the calls to `fireline_manager.update()`
        and `fire_map.update()` pass tests.
        '''
        # Check against a completely burned fire_map
        fire_map = np.full((self.config.area.screen_size, self.config.area.screen_size),
                           BurnStatus.BURNED)

        self.fire_map = self.simulation_flat.run(time='1h')
        # assert the fire map is all BURNED
        self.assertEqual(
            self.fire_map.max(),
            fire_map.max(),
            msg=f'The fire map has a maximum BurnStatus of {self.fire_map.max()} '
            f', but it should be {fire_map.max()}')

        self.simulation_flat.reset()

        # Check that we can run for one step
        self.fire_map = self.simulation_flat.run(time=1)
        self.assertEqual(self.simulation_flat.elapsed_time,
                         self.config.simulation.update_rate,
                         msg=f'Only {self.config.simulation.update_rate}m should  '
                             f'passed, but {self.simulation_flat.elapsed_time}m has '
                             'passed.')

    def test__render_inline(self) -> None:
        '''
        Test that the call to `_render_inline()` runs through properly.
        '''
        self.config.render.inline = True
        # set position array
        current_agent_loc = np.zeros(
            (self.config.area.screen_size, self.config.area.screen_size))
        loc = (1, 2)
        current_agent_loc[loc] = 1

        # set correct mitigation array
        mitigation = np.zeros(
            (self.config.area.screen_size, self.config.area.screen_size))
        mit_point = (1, 1)
        mitigation[mit_point] = BurnStatus.FIRELINE
        self.config.simulation.headless = True
        # Test rendering 'inline' (as agent traverses)
        self.simulation._render_inline(mitigation, current_agent_loc)
        # assert the points are placed
        self.assertEqual(self.simulation.fireline_manager.sprites[0].pos,
                         mit_point,
                         msg=(f'The position of the sprite is '
                              f'{self.simulation.fireline_manager.sprites[0].pos} '
                              f', but it should be {mit_point}'))

    def test__render_mitigations(self) -> None:
        '''

        '''
        mitigation = np.full((self.config.area.screen_size, self.config.area.screen_size),
                             BurnStatus.FIRELINE)

        self.config.simulation.headless = True
        self.simulation._render_mitigations(mitigation)
        full_grid = self.config.area.screen_size * self.config.area.screen_size
        self.assertEqual(
            len(self.simulation.fireline_sprites),
            full_grid,
            msg=f'The total number of mitigated pixels should be {full_grid} '
            f'but are actually {len(self.simulation.fireline_sprites)}')

    def test__render_mitigation_fire_spread(self) -> None:
        '''

        '''
        # assert the points are placed and fire can spread
        self.fireline_sprites = self.simulation.fireline_sprites_empty

        mitigation = np.zeros(
            (self.config.area.screen_size, self.config.area.screen_size))
        # start the fire where we have a control line
        mitigation[self.config.fire.fire_initial_position[0] - 1:] = 3
        self.config.mitigation.ros_attenuation = False
        self.config.simulation.headless = True
        self.simulation._render_mitigation_fire_spread(mitigation)

        # assert no fire has spread
        self.assertTrue(len(self.simulation.fire_sprites) == 1,
                        msg=f'The returned state of the Game should have no fire spread, '
                        f' but, has {len(self.simulation.fire_sprites)}.')

    def test_render(self) -> None:
        '''
        Test that the call to `_render()` runs through properly.

        This should be pass as long as the calls to `fireline_manager.update()` and
        `fire_map.update()` pass tests.

        This should pass as long as the calls to `_render_inline`,
            `_render_mitigations`, and `_render_mitigation_fire_spread` pass.

        Assert the points get updated in the `fireline_sprites` group.

        '''
        pass

    def test_get_seeds(self) -> None:
        '''
        Test the get_seeds method and ensure it returns all available seeds
        '''
        seeds = self.simulation.get_seeds()
        flat_seeds = self.simulation_flat.get_seeds()

        for key, seed in seeds.items():
            msg = (f'The seed for {key} ({seed}) does not match that found in '
                   '{self.config.path}')
            if key == 'elevation':
                self.assertEqual(seed, self.config.terrain.perlin.seed, msg=msg)
            if key == 'fuel':
                self.assertEqual(seed, self.config.terrain.chaparral.seed, msg=msg)
            if key == 'wind_speed':
                self.assertEqual(seed, self.config.wind.perlin.speed.seed, msg=msg)
            if key == 'wind_direction':
                self.assertEqual(seed, self.config.wind.perlin.direction.seed, msg=msg)

        # Test for different use-cases where not all functions have seeds
        self.assertNotIn('elevation', flat_seeds)
        self.assertNotIn('wind_speed', flat_seeds)
        self.assertNotIn('wind_direction', flat_seeds)

        for key, seed in flat_seeds.items():
            msg = (f'The seed for {key} ({seed}) does not match that found in '
                   '{self.config.path}')
            if key == 'fuel':
                self.assertEqual(seed,
                                 self.config_flat_simple.terrain.chaparral.seed,
                                 msg=msg)

    def test_set_seeds(self) -> None:
        '''
        Test the set_seeds method and ensure it re-instantiates the required functions
        '''
        seed = 1234
        seeds = {
            'elevation': seed,
            'fuel': seed,
            'wind_speed': seed,
            'wind_direction': seed
        }
        self.simulation.set_seeds(seeds)
        returned_seeds = self.simulation.get_seeds()

        self.assertEqual(seeds,
                         returned_seeds,
                         msg=f'The input seeds ({seeds}) do not match the returned seeds '
                         f'({returned_seeds})')

        # Only set wind_speed and not wind_direction
        seed = 2345
        seeds = {'elevation': seed, 'fuel': seed, 'wind_speed': seed}
        self.simulation.set_seeds(seeds)
        returned_seeds = self.simulation.get_seeds()

        # Put the previous value for wind_direction into the dictionary so we can check
        # to make sure it wasn't changed
        seeds['wind_direction'] = 1234
        self.assertEqual(seeds,
                         returned_seeds,
                         msg=f'The input seeds ({seeds}) do not match the returned seeds '
                         f'({returned_seeds})')

        # Only set wind_direction and not wind_speed
        seed = 3456
        seeds = {'wind_direction': seed}
        self.simulation.set_seeds(seeds)
        returned_seeds = self.simulation.get_seeds()

        # Put the previous value for wind_direction into the dictionary so we can check
        # to make sure it wasn't changed
        seeds['elevation'] = 2345
        seeds['fuel'] = 2345
        seeds['wind_speed'] = 2345
        self.assertEqual(seeds,
                         returned_seeds,
                         msg=f'The input seeds ({seeds}) do not match the returned seeds '
                         f'({returned_seeds})')

        # Give no valid keys to hit the log warning
        seeds = {'not_valid': 1111}
        success = self.simulation.set_seeds(seeds)
        self.assertFalse(success,
                         msg='The set_seeds method should have returned False '
                         f'with input seeds set to {seeds}')
