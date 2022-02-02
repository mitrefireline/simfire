from typing import Dict
import unittest
import numpy as np
from ...utils.config import Config
from ...rl_env.simulation import RothermelSimulation
from ...enums import BurnStatus


class RothermelSimulationTest(unittest.TestCase):
    def setUp(self) -> None:
        '''
        '''
        self.config = Config('./src/utils/_tests/test_configs/test_config.yml')
        self.simulation = RothermelSimulation(self.config)

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

    def test_create_fire(self) -> None:
        '''
        Test that the fire (FireManager) gets created properly.
            This should work as long as the tests for FireManager() work.
        '''
        pass

    def test_get_actions(self) -> None:
        '''
        Test that the call to get_actions() runs properly and returns all Rothermel
            FireLineManager() features.
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

        # assert points get updated after agent traverses entire game
        self.config.render.inline = False
        self.config.render.post_agent = True
        current_agent_loc = (self.config.area.screen_size, self.config.area.screen_size)
        self.mitigation = np.full(
            (self.config.area.screen_size, self.config.area.screen_size),
            BurnStatus.FIRELINE)
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
        mitigation = np.zeros(
            (self.config.area.screen_size, self.config.area.screen_size))
        mitigation[1, 0] = 1
        position = np.zeros((self.config.area.screen_size, self.config.area.screen_size))
        position[self.config.area.screen_size - 1, self.config.area.screen_size - 1] = 1

        fire_map = np.full((self.config.area.screen_size, self.config.area.screen_size),
                           BurnStatus.BURNED)

        self.fire_map = self.simulation.run(mitigation, False)
        # assert the fire map is all BURNED
        self.assertEqual(
            self.fire_map.max(),
            fire_map.max(),
            msg=f'The fire map has a maximum BurnStatus of {self.fire_map.max()} '
            f', but it should be {fire_map.max()}')

        # assert fire map has BURNED and FIRELINE pixels
        fire_map[1, 0] = 3
        self.fire_map = self.simulation.run(mitigation, True)
        self.assertEqual(len(np.where(self.fire_map == 3)),
                         len(np.where(fire_map == 1)),
                         msg=f'The fire map has a mitigation sprite of length '
                         f'{len(np.where(self.fire_map == 3))}, but it should be '
                         f'{len(np.where(fire_map == 1))}')

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
        self.config.simulation.headless = False
        self.simulation = RothermelSimulation(self.config)
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

        self.config.simulation.headless = False
        self.simulation = RothermelSimulation(self.config)
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
        self.config.simulation.headless = False
        self.simulation = RothermelSimulation(self.config)
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
