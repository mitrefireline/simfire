from typing import Dict
import unittest
import numpy as np
from ...utils.config import Config
from ...rl_env.simulation import RothermalSimulation
from ...enums import BurnStatus, GameStatus


class RothermalSimulationTest(unittest.TestCase):
    def setUp(self) -> None:
        '''
        '''
        self.config = Config('./src/rl_env/_tests/test_config.yml')
        self.simulation = RothermalSimulation(self.config)

    def test__create_terrain(self) -> None:
        '''
        Test that the terrain gets created properly. This should work as long
            as the tests for Terrain() work.
        '''
        pass

    def test__create_mitigations(self) -> None:
        '''
        Test that the mitigation (FireLineManager) gets created properly.
            This should work as long as the tests for FireLineManager() work.
        '''
        pass

    def test__create_wind(self) -> None:
        '''
        Test that the wind properties gets created properly. This should work as long
            as the tests for Wind() work.
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
        Test that the call to get_actions() runs properly and returns all Rothermal
            FireLineManager() features.
        '''
        simulation_actions = self.simulation.get_actions()
        self.assertIsInstance(simulation_actions, Dict)

    def test_get_attributes(self) -> None:
        '''
        Test that the call to get_actions() runs properly and returns all Rothermal
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
        current_agent_loc = (1, 1)
        self.mitigation = BurnStatus.FIRELINE
        points = set([current_agent_loc])
        self.simulation._update_sprite_points(self.mitigation,
                                              current_agent_loc,
                                              inline=True)
        self.assertEqual(self.simulation.points,
                         points,
                         msg=f'The sprite was updated at {self.simulation.points}, '
                         f'but it should have been at {current_agent_loc}')

        # assert points get updated after agent traverses entire game
        current_agent_loc = (self.config.area.screen_size, self.config.area.screen_size)
        self.mitigation = np.full(
            (self.config.area.screen_size, self.config.area.screen_size),
            BurnStatus.FIRELINE)
        points = [(i, j) for j in range(self.config.area.screen_size)
                  for i in range(self.config.area.screen_size)]
        points = set(points)
        self.simulation._update_sprite_points(self.mitigation,
                                              current_agent_loc,
                                              inline=False)
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
        mitigation = np.zeros(
            (self.config.area.screen_size, self.config.area.screen_size))
        mitigation[1, 0] = 1
        position = np.zeros((self.config.area.screen_size, self.config.area.screen_size))
        position[self.config.area.screen_size - 1, self.config.area.screen_size - 1] = 1

        fire_map = np.full((self.config.area.screen_size, self.config.area.screen_size),
                           BurnStatus.BURNED)

        self.fire_map = self.simulation.run(mitigation, position, False)
        # assert the fire map is all BURNED
        self.assertEqual(
            self.fire_map.max(),
            fire_map.max(),
            msg=f'The fire map has a maximum BurnStatus of {self.fire_map.max()} '
            f', but it should be {fire_map.max()}')

        # assert fire map has BURNED and FIRELINE pixels
        fire_map[1, 0] = 3
        self.fire_map = self.simulation.run(mitigation, position, True)
        self.assertEqual(len(np.where(self.fire_map == 3)),
                         len(np.where(fire_map == 1)),
                         msg=f'The fire map has a mitigation sprite of length '
                         f'{len(np.where(self.fire_map == 3))}, but it should be '
                         f'{len(np.where(fire_map == 1))}')

    def test_render(self) -> None:
        '''
        Test that the call to `_render()` runs through properly.

        This should be pass as long as the calls to `fireline_manager.update()` and
        `fire_map.update()` pass tests.

        Assert the points get updated in the `fireline_sprites` group.

        '''
        current_agent_loc = (1, 1)

        # Test rendering 'inline' (as agent traverses)
        self.simulation.render(BurnStatus.FIRELINE, current_agent_loc, inline=True)
        # assert the points are placed
        self.assertEqual(self.simulation.fireline_manager.sprites[0].pos,
                         current_agent_loc,
                         msg=(f'The position of the sprite is '
                              f'{self.simulation.fireline_manager.sprites[0].pos} '
                              f', but it should be {current_agent_loc}'))

        # Test Full Mitigation (after agent traversal)
        self.fireline_sprites = self.simulation.fireline_sprites_empty
        mitigation = np.full((self.config.area.screen_size, self.config.area.screen_size),
                             BurnStatus.FIRELINE)
        self.simulation.render(
            mitigation, (self.config.area.screen_size, self.config.area.screen_size))
        # assert the points are placed
        self.assertEqual(len(self.simulation.fireline_manager.sprites),
                         self.config.area.screen_size**2 + 1,
                         msg=(f'The number of sprites updated is '
                              f'{len(self.simulation.fireline_manager.sprites)} '
                              f', but it should be {self.config.area.screen_size**2+1}'))

        # Test Full Mitigation (after agent traversal) and fire spread

        # assert the points are placed and fire can spread
        self.fireline_sprites = self.simulation.fireline_sprites_empty

        mitigation = np.zeros(
            (self.config.area.screen_size, self.config.area.screen_size))
        # start the fire where we have a control line
        mitigation[self.config.fire.fire_initial_position[0] - 1:] = 1
        self.simulation.render(
            mitigation, (self.config.area.screen_size, self.config.area.screen_size),
            mitigation_only=False,
            mitigation_and_fire_spread=True)

        self.assertEqual(
            self.simulation.fire_status,
            GameStatus.QUIT,
            msg=f'The returned state of the Game is {self.simulation.game_status} '
            ' but, should be GameStatus.QUIT.')
