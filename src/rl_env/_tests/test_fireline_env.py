import unittest
import numpy as np

from . import test_config as config
from ...game.sprites import Terrain
from ...game.game import Game
from ...enums import BurnStatus
from ...world.parameters import Environment, FuelArray, FuelParticle, Tile
from ...game.managers.fire import RothermelFireManager
from ...game.managers.mitigation import FireLineManager
from ...rl_env.fireline_env import FireLineEnv, RLEnv


class RLEnvTest(unittest.TestCase):
    def setUp(self) -> None:
        '''
        Initialize the FireLineEnv class and instantiate a fireline action.

        '''
        self.fireline_env = FireLineEnv(config)
        self.rl_env = RLEnv(self.fireline_env)
        self.action = 1
        self.current_agent_loc = (1, 1)

    def test_init(self) -> None:
        '''
        Test setting the seed for the terrain map.

        '''
        seed = 1212
        fireline_env_seed = FireLineEnv(config, seed)
        fireline_env_seed = fireline_env_seed.config.terrain_map
        fireline_env_no_seed = FireLineEnv(config)
        fireline_env_no_seed = fireline_env_no_seed.config.terrain_map

        # assert these envs are different
        for i, j in zip(fireline_env_seed, fireline_env_no_seed):
            for fuel_i, fuel_j in zip(i, j):
                self.assertNotEqual(
                    fuel_i.w_0,
                    fuel_j.w_0,
                    msg='Different seeds should produce different terrain '
                    'maps.')

        # assert equal Fuel Maps
        fireline_env_same_seed = FireLineEnv(config, seed)
        fireline_env_same_seed = fireline_env_same_seed.config.terrain_map
        self.assertEqual(fireline_env_seed,
                         fireline_env_same_seed,
                         msg='Same seeds should produce the same terrain '
                         'maps.')

    def test_step(self) -> None:
        '''
        Test that the call to step() runs through properly.
        Step() calls a handful of sub-functions, which also get tested.

        TODO: This will change with updates to the state format
        '''
        self.rl_env.reset()
        state, reward, done, _ = self.rl_env.step(self.action)

        agent_pos = np.where(state[0] == 1)

        self.assertEqual(
            agent_pos, (0, 1),
            msg=(f'The returned state of agent position of the game is {agent_pos}, '
                 f'but it should be (1, 0)'))

        self.assertEqual(
            reward,
            0,
            msg=(f'The returned state of reward of the game is {reward}, but it '
                 f'should be -1'))

        self.assertEqual(done,
                         False,
                         msg=(f'The returned state of the game is {done}, but it '
                              f'should be False'))

    def test_reset(self) -> None:
        '''
        Test that the call to reset() runs through properly.
        reset() calls a handful of sub-functions, which also get tested.
        Assert agent position is returned to upper left corner (1,0) of game.

        state_space:

        0:      'position'
        1:      'terrain: w_0'
        2:      'elevation'
        3:      'mitigation'
        '''
        state = self.rl_env.reset()

        agent_pos = np.where(state[0] == 1)
        fuel_arrays = state[1]
        fireline = state[-1].max()
        elevation = state[2]

        w_0_array = np.array([
            self.fireline_env.terrain.fuel_arrs[i][j].fuel.w_0
            for j in range(self.fireline_env.config.screen_size)
            for i in range(self.fireline_env.config.screen_size)
        ]).reshape(self.fireline_env.config.screen_size,
                   self.fireline_env.config.screen_size)

        self.assertEqual(
            agent_pos, (0, 0),
            msg=(f'The returned state of the agent position is {agent_pos}, but it '
                 f'should be [0, 0]'))

        self.assertTrue(
            (fuel_arrays == w_0_array).all(),
            msg=(f'The returned state of the fuel arrays is {fuel_arrays}, but it '
                 f'should be 1'))

        self.assertEqual(fireline,
                         0,
                         msg=(f'The returned state of the fireline is {fireline}, but it '
                              f'should be 0'))

        self.assertTrue(
            (elevation == (self.fireline_env.terrain.elevations +
                           self.fireline_env.config.noise_amplitude) /
             (2 * self.fireline_env.config.noise_amplitude)).all(),
            msg=('The returned state of the terrain elevation map is not the same '
                 'as the initialized terrain elevation map'))

    def test_update_current_agent_loc(self) -> None:
        '''
        Test that the call to _update_current_agent_loc() runs through properly.
        The agent position should only be at a single location per step().
        '''

        self.rl_env.reset()
        self.rl_env.current_agent_loc = self.current_agent_loc
        x = self.current_agent_loc[0]
        y = self.current_agent_loc[1]
        new_agent_loc = (x, y + 1)
        self.rl_env._update_current_agent_loc()

        self.assertEqual(
            self.rl_env.current_agent_loc,
            new_agent_loc,
            msg=f'The returned agent location is {self.current_agent_loc}, but it '
            f'should be {new_agent_loc}.')


class FireLineEnvTest(unittest.TestCase):
    def setUp(self) -> None:
        '''
        Initialize the FireLineEnv class and instantiate a fireline action.

        '''
        self.config = config
        self.fireline_env = FireLineEnv(self.config)
        self.mitigation = True
        self.fire_spread = False

        self.game = Game(self.config.screen_size)
        self.fuel_particle = FuelParticle()
        self.fuel_arrs = [[
            FuelArray(Tile(j, i, config.terrain_scale, config.terrain_scale),
                      config.terrain_map[i][j]) for j in range(self.config.terrain_size)
        ] for i in range(self.config.terrain_size)]
        self.terrain = Terrain(self.fuel_arrs, self.config.elevation_fn,
                               self.config.terrain_size, self.config.screen_size)
        self.environment = Environment(self.config.M_f, self.config.U, self.config.U_dir)

        # initialize all mitigation strategies
        self.fireline_manager = FireLineManager(size=self.config.control_line_size,
                                                pixel_scale=self.config.pixel_scale,
                                                terrain=self.terrain)
        self.fireline_sprites = self.fireline_manager.sprites
        self.fireline_sprites_reset = self.fireline_manager.sprites.copy()
        self.fire_manager = RothermelFireManager(
            self.config.fire_init_pos, self.config.fire_size,
            self.config.max_fire_duration, self.config.pixel_scale,
            self.config.update_rate, self.fuel_particle, self.terrain, self.environment)
        self.fire_sprites = self.fire_manager.sprites

    def test_render(self) -> None:
        '''
        Test that the call to _render() runs through properly.
        This should be pass as long as the calls to fireline_manager.update()
        and fire_map.update() pass tests, respectively.
        Assert the points get updated in the fireline_sprites group.

        TODO: FIX - Putting a control_line "down" before spreading fire, the fire_map
                changes from GameStatus.FIRELINE -> GameStatus.BURNED
        '''
        current_agent_loc = (1, 1)

        # Test rendering 'inline' (as agent traverses)
        self.fireline_env._render(1, current_agent_loc, inline=True)
        # assert the points are placed
        self.assertEqual(self.fireline_env.fireline_manager.sprites[0].pos,
                         current_agent_loc,
                         msg=(f'The position of the sprite is '
                              f'{self.fireline_env.fireline_manager.sprites[0].pos} '
                              f', but it should be {current_agent_loc}'))

        # Test Full Mitigation (after agent traversal)
        self.fireline_sprites = self.fireline_sprites_reset
        mitigation = np.full((self.config.screen_size, self.config.screen_size), 1)
        self.fireline_env._render(mitigation,
                                  (self.config.screen_size, self.config.screen_size))
        # assert the points are placed
        self.assertEqual(len(self.fireline_env.fireline_manager.sprites),
                         self.config.screen_size**2 + 1,
                         msg=(f'The number of sprites updated is '
                              f'{len(self.fireline_env.fireline_manager.sprites)} '
                              f', but it should be {self.config.screen_size**2+1}'))

        # Test Full Mitigation (after agent traversal) and fire spread

        # assert the points are placed and fire can spread
        self.fireline_sprites = self.fireline_sprites_reset

        mitigation = np.zeros((self.config.screen_size, self.config.screen_size))
        # start the fire where we have a control line
        mitigation[self.config.fire_init_pos[0] - 1:] = 1
        self.fireline_env._render(mitigation,
                                  (self.config.screen_size, self.config.screen_size),
                                  mitigation_only=False,
                                  mitigation_and_fire_spread=True)

        all_burning_locs = list(zip(*np.where(self.fireline_env.fire_map == 2)))

        self.assertIn(self.config.fire_init_pos,
                      all_burning_locs,
                      msg=(f'The number of sprites updated is '
                           f'{len(np.where(self.fireline_env.fire_map == 2))} '
                           f', but it should be {self.config.fire_init_pos}'))

    def test_update_sprite_points(self) -> None:
        '''
        Test that the call to _update_sprites() runs through properly.
        Since self.action is instantiated as 1, we need to verify that
            a fireline sprite is created and added to the fireline_manager.
        '''

        # assert points get updated 'inline' as agent traverses
        current_agent_loc = (1, 1)
        self.mitigation = 1
        points = set([current_agent_loc])
        self.fireline_env._update_sprite_points(self.mitigation,
                                                current_agent_loc,
                                                inline=True)
        self.assertEqual(self.fireline_env.points,
                         points,
                         msg=f'The sprite was updated at {self.fireline_env.points} '
                         f', but it should have been at {current_agent_loc}')

        # assert points get updated after agent traverses entire game
        current_agent_loc = (self.config.screen_size, self.config.screen_size)
        self.mitigation = np.full((self.config.screen_size, self.config.screen_size), 1)
        points = [(i, j) for j in range(self.config.screen_size)
                  for i in range(self.config.screen_size)]
        points = set(points)
        self.fireline_env._update_sprite_points(self.mitigation,
                                                current_agent_loc,
                                                inline=False)
        self.assertEqual(
            self.fireline_env.points,
            points,
            msg=f'The number of sprites updated was {len(self.fireline_env.points)} '
            f', but it should have been {len(points)} sprites.')

    def test_reset_state(self) -> None:
        '''
        Test that the call to _convert_data_to_gym runs through properly.
        This function returns the state as an array.

        TODO: Waiting for update to the state space

        '''

    def test_run(self) -> None:
        '''
        Test that the call to _run runs the simulation properly.
        This function returns the burned firemap with or w/o mitigation.

        This function will reset the fire_map to all UNBURNED pixels
            at each call to the method.

        This should pass as long as the calls to fireline_manager.update()
        and fire_map.update() pass tests, respectively.
        '''
        mitigation = np.zeros((self.config.screen_size, self.config.screen_size))
        mitigation[1, 0] = 1
        position = np.zeros((self.config.screen_size, self.config.screen_size))
        position[self.config.screen_size - 1, self.config.screen_size - 1] = 1

        fire_map = np.full((self.config.screen_size, self.config.screen_size),
                           BurnStatus.BURNED)

        self.fire_map = self.fireline_env._run(mitigation, position, False)
        # assert the fire map is all BURNED
        self.assertEqual(
            self.fire_map.max(),
            fire_map.max(),
            msg=f'The fire map has a maximum BurnStatus of {self.fire_map.max()} '
            f', but it should be {fire_map.max()}')

        # assert fire map has BURNED and FIRELINE pixels
        fire_map[1, 0] = 3
        self.fire_map = self.fireline_env._run(mitigation, position, True)
        self.assertEqual(len(np.where(self.fire_map == 3)),
                         len(np.where(fire_map == 1)),
                         msg=f'The fire map has a mitigation sprite of length '
                         f'{len(np.where(self.fire_map == 3))}, but it should be '
                         f'{len(np.where(fire_map == 1))}')

    def test_compare_states(self) -> None:
        '''
        Test that the call to _compare_states runs the comparison of
            state spaces properly.
        This function returns the overall reward.
        '''
        screen_size = self.fireline_env.config.screen_size
        # create array of BURNED pixels
        fire_map = np.full((screen_size, screen_size), 2)
        # create array of agent mitigation + fire spread (BURNED pixels)
        fire_map_with_agent = np.full((screen_size, screen_size), 3)

        unmodified_reward = -1 * (screen_size * screen_size)
        modified_reward = -1 * (screen_size * screen_size)
        test_reward = (modified_reward - unmodified_reward) / \
                        (screen_size * screen_size)
        reward = self.fireline_env._compare_states(fire_map, fire_map_with_agent)

        # assert rewards are the same
        self.assertEqual(reward,
                         test_reward,
                         msg=(f'The returned reward of the game is {reward}, but it '
                              f'should be {test_reward}'))
