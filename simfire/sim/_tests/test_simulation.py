import os
import unittest
from pathlib import Path
from typing import Dict

import numpy as np

from ...enums import BurnStatus
from ...game.sprites import Agent, Terrain
from ...sim.simulation import FireSimulation
from ...utils.config import Config, ConfigError
from ...utils.log import create_logger

log = create_logger(__name__)

os.environ["SDL_VIDEODRIVER"] = "dummy"


class FireSimulationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config("./simfire/utils/_tests/test_configs/test_config.yml")
        self.config_flat_simple = Config(
            Path(self.config.path).parent / "test_config_flat_simple.yml"
        )

        self.screen_size = (self.config.area.screen_size, self.config.area.screen_size)

        self.simulation = FireSimulation(self.config)
        self.simulation_flat = FireSimulation(self.config_flat_simple)

        topo_layer = self.config.terrain.topography_layer
        fuel_layer = self.config.terrain.fuel_layer
        self.terrain = Terrain(fuel_layer, topo_layer, self.screen_size)

    def test__create_terrain(self) -> None:
        """
        Test that the terrain gets created properly.

        This should work as long as the tests for Terrain() work.
        """
        pass

    def test__create_mitigations(self) -> None:
        """
        Test that the mitigation (FireLineManager) gets created properly.

        This should work as long as the tests for FireLineManager() work.
        """
        pass

    def test_create_fire(self) -> None:
        """
        Test that the fire (FireManager) gets created properly.

        This should work as long as the tests for FireManager() work.
        """
        pass

    def test_get_actions(self) -> None:
        """
        Test that the call to `get_actions()` runs properly and returns all fire
        `FireLineManager()` features.
        """
        simulation_actions = self.simulation.get_actions()
        self.assertIsInstance(simulation_actions, Dict)

    def test_get_attribute_bounds(self) -> None:
        """
        Test that the call to get_actions() runs properly and returns all fire
        features (Fire, Wind, FireLine, Terrain).
        """
        simulation_attributes = self.simulation.get_attribute_bounds()
        self.assertIsInstance(simulation_attributes, Dict)

    def test_get_attribute_data(self) -> None:
        """
        Test that the call to get_actions() runs properly and returns all fire
        features (Fire, Wind, FireLine, Terrain).
        """
        simulation_attributes = self.simulation.get_attribute_data()
        self.assertIsInstance(simulation_attributes, Dict)

    def test_run(self) -> None:
        """
        Test that the call to `_run` runs the simulation properly.

        This function returns the burned firemap with or w/o mitigation.

        This function will reset the `fire_map` to all `UNBURNED` pixels at each call to
        the method.

        This should pass as long as the calls to `fireline_manager.update()`
        and `fire_map.update()` pass tests.
        """
        # Check against a completely burned fire_map
        fire_map = np.full(
            (self.config.area.screen_size, self.config.area.screen_size),
            BurnStatus.BURNED,
        )

        self.fire_map, _ = self.simulation_flat.run(time="1h")
        # assert the fire map is all BURNED
        self.assertEqual(
            self.fire_map.max(),
            fire_map.max(),
            msg=f"The fire map has a maximum BurnStatus of {self.fire_map.max()} "
            f", but it should be {fire_map.max()}",
        )

        self.simulation_flat.reset()

        # Check that we can run for one step
        self.fire_map, _ = self.simulation_flat.run(time=1)
        self.assertEqual(
            self.simulation_flat.elapsed_time,
            self.config.simulation.update_rate,
            msg=f"Only {self.config.simulation.update_rate}m should  "
            f"passed, but {self.simulation_flat.elapsed_time}m has "
            "passed.",
        )

    def test_get_seeds(self) -> None:
        """
        Test the get_seeds method and ensure it returns all available seeds
        """
        seeds = self.simulation.get_seeds()
        flat_seeds = self.simulation_flat.get_seeds()

        for key, seed in seeds.items():
            msg = (
                f"The seed for {key} ({seed}) does not match that found in "
                "{self.config.path}"
            )
            if key == "elevation":
                self.assertEqual(
                    seed, self.config.terrain.topography_function.kwargs["seed"], msg=msg
                )
            if key == "fuel":
                self.assertEqual(
                    seed, self.config.terrain.fuel_function.kwargs["seed"], msg=msg
                )
            if key == "wind_speed":
                self.assertEqual(
                    seed, self.config.wind.speed_function.kwargs["seed"], msg=msg
                )
            if key == "wind_direction":
                self.assertEqual(
                    seed, self.config.wind.direction_function.kwargs["seed"], msg=msg
                )

        # Test for different use-cases where not all functions have seeds
        self.assertNotIn("elevation", flat_seeds)
        self.assertNotIn("wind_speed", flat_seeds)
        self.assertNotIn("wind_direction", flat_seeds)

        for key, seed in flat_seeds.items():
            msg = (
                f"The seed for {key} ({seed}) does not match that found in "
                f"{self.config.path}"
            )
            if key == "fuel":
                cfg_seed = self.config_flat_simple.terrain.fuel_function.kwargs["seed"]
                self.assertEqual(seed, cfg_seed, msg=msg)

    def test_set_seeds(self) -> None:
        """
        Test the set_seeds method and ensure it re-instantiates the required functions
        """
        seed = 1234
        seeds = {
            "elevation": seed,
            "fuel": seed,
            "wind_speed": seed,
            "wind_direction": seed,
            "fire_initial_position": seed,
        }
        self.simulation.set_seeds(seeds)
        returned_seeds = self.simulation.get_seeds()

        self.assertDictEqual(
            seeds,
            returned_seeds,
            msg=f"The input seeds ({seeds}) do not match the returned seeds "
            f"({returned_seeds})",
        )

        # Only set wind_speed and not wind_direction
        seed = 2345
        seeds = {
            "elevation": seed,
            "fuel": seed,
            "wind_speed": seed,
            "fire_initial_position": seed,
        }
        self.simulation.set_seeds(seeds)
        returned_seeds = self.simulation.get_seeds()

        # Put the previous value for wind_direction into the dictionary so we can check
        # to make sure it wasn't changed
        seeds["wind_direction"] = 1234
        self.assertDictEqual(
            seeds,
            returned_seeds,
            msg=f"The input seeds ({seeds}) do not match the returned seeds "
            f"({returned_seeds})",
        )

        # Only set wind_direction and not wind_speed
        seed = 3456
        seeds = {"wind_direction": seed}
        self.simulation.set_seeds(seeds)
        returned_seeds = self.simulation.get_seeds()

        # Put the previous value for wind_direction into the dictionary so we can check
        # to make sure it wasn't changed
        seeds["elevation"] = 2345
        seeds["fuel"] = 2345
        seeds["wind_speed"] = 2345
        seeds["fire_initial_position"] = 2345
        self.assertDictEqual(
            seeds,
            returned_seeds,
            msg=f"The input seeds ({seeds}) do not match the returned seeds "
            f"({returned_seeds})",
        )

        # Test the fire initial position seed
        seed = 208
        seeds = {"fire_initial_position": seed}
        self.simulation.set_seeds(seeds)
        returned_seed = self.simulation.get_seeds()["fire_initial_position"]
        self.assertEqual(
            seed,
            returned_seed,
            msg=f"The input fire initial position seed ({seed}) does not "
            f"match the returned seed ({returned_seed})",
        )

        # Give no valid keys to hit the log warning
        seeds = {"not_valid": 1111}
        success = self.simulation.set_seeds(seeds)
        self.assertFalse(
            success,
            msg="The set_seeds method should have returned False "
            f"with input seeds set to {seeds}",
        )

    def test_get_layer_types(self) -> None:
        """
        Test the getting of the layer types
        """
        layer_types = self.simulation.get_layer_types()
        self.assertIsInstance(layer_types, Dict)
        self.assertIn("elevation", layer_types)
        self.assertIn("fuel", layer_types)

    def test_set_layer_types(self) -> None:
        """
        Test setting the layer types
        """
        layer_types = {"elevation": "operational", "fuel": "operational"}
        self.simulation.set_layer_types(layer_types)
        self.assertDictEqual(layer_types, self.simulation.get_layer_types())

        # Test that the layer types are set to the default if the input is not valid
        layer_types = {"elevation": "not_valid", "fuel": "operational"}
        self.assertRaises(ConfigError, self.simulation.set_layer_types, layer_types)

        # Test that we output a log warning
        layer_types = {"asdf": "functional", "qwer": "functional"}
        self.assertWarns(Warning, self.simulation.set_layer_types, layer_types)

    def test_load_mitigation(self) -> None:
        """
        Test loading a mitigation map
        """
        old_map = np.copy(self.simulation.fire_map)

        new_map = np.zeros((9, 9))
        new_map[0][0] = 10

        self.assertWarns(Warning, self.simulation.load_mitigation, new_map)
        self.assertTrue(np.array_equal(old_map, self.simulation.fire_map))

        new_map[0][0] = 3
        self.simulation.load_mitigation(new_map)

        self.assertTrue(np.array_equal(new_map, self.simulation.fire_map))

    def test_get_disaster_categories(self) -> None:
        """
        Test getting all possible categories a pixel can be
        """
        categories = self.simulation.get_disaster_categories()
        self.assertEqual(list(categories.keys()), list(BurnStatus.__members__))
        self.assertEqual(list(categories.values()), [e.value for e in BurnStatus])

    def test__create_out_path(self) -> None:
        """
        Test the creation of the output path
        """
        out_path = self.simulation._create_out_path()
        self.assertIsInstance(out_path, Path)
        out_path.rmdir()

    def test_update_agent_positions(self) -> None:
        """
        Test the update_agent_positions method
        """
        self.simulation.update_agent_positions([(1, 1, 5)])
        self.assertEqual(
            self.simulation.agent_positions[1][1],
            5,
            msg="The agent ID was " "not set to the correct value",
        )
        self.assertIsInstance(self.simulation.agents[5], Agent)
        self.simulation.update_agent_positions([(2, 2, 5)])
        self.assertEqual(
            self.simulation.agent_positions[2][2],
            5,
            msg="The agent was " "not moved correctly",
        )
        self.assertNotEqual(
            self.simulation.agent_positions[1][1],
            5,
            msg="The agent was " "not removed from the original position",
        )

    def test__create_agent_positions(self) -> None:
        """
        Test creating the agent positions
        """
        agent_positions = np.zeros_like(self.simulation.fire_map)
        self.assertTrue(
            self.simulation.agent_positions.all() == agent_positions.all(),
            msg="The agent positions are not the same as the initial fire "
            "map (all zeros)",
        )

    def test_rendering(self) -> None:
        """
        Test setting the `rendering` property
        """
        self.simulation.rendering = True
        self.assertTrue(
            self.simulation.rendering, msg="`rendering` was not set correctly to True"
        )
        self.assertIsNotNone(self.simulation._game, msg="simulation._game was not set")
        self.simulation.rendering = False
        self.assertFalse(
            self.simulation.rendering, msg="simulation.rendering was not set to False"
        )

    def test_save_gif(self) -> None:
        """
        Test the saving of the GIF after running

        In the config, headless must be set to false for this test to pass
        """
        self.simulation_flat.config.simulation.headless = False
        self.simulation_flat.rendering = True
        self.simulation_flat.run(1)
        self.simulation_flat.save_gif("tmp.gif")
        tmp_file = Path("tmp.gif")
        self.assertTrue(tmp_file.exists(), msg="The GIF was not saved correctly")
        tmp_file.unlink()
        self.simulation_flat.save_gif("tmp")
        tmp_file = Path("tmp")
        self.assertTrue(tmp_file.is_dir(), msg="The GIF was not saved correctly")
        [f.unlink() for f in tmp_file.iterdir()]
        tmp_file.rmdir()
        self.simulation_flat.save_gif("tmp/tmp_0.gif")
        tmp_file = Path("tmp/tmp_0.gif")
        self.assertTrue(tmp_file.exists(), msg="The GIF was not saved correctly")
        tmp_file.unlink()
        tmp_file.parent.rmdir()
        self.simulation_flat.rendering = False

    def test_save_spread_graph(self) -> None:
        """
        Test the saving of the spread graph

        In the config, headless must be set to false for this test to pass
        """
        self.simulation_flat.config.simulation.headless = False
        self.simulation_flat.rendering = True
        self.simulation_flat.run(1)
        self.simulation_flat.save_spread_graph("tmp.png")
        tmp_file = self.simulation_flat._create_out_path() / "tmp.png"
        self.assertTrue(tmp_file.exists(), msg="The spread graph was not saved correctly")
        tmp_file.unlink()
        self.simulation_flat.save_spread_graph("tmp")
        self.assertTrue(tmp_file.exists(), msg="The spread graph was not saved correctly")
        tmp_file.unlink()
        tmp_file.parent.rmdir()
        self.simulation_flat.rendering = False
