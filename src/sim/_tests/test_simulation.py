import unittest
from pathlib import Path
from typing import Dict

import numpy as np

from ...enums import BurnStatus
from ...game.sprites import Terrain
from ...sim.simulation import RothermelSimulation
from ...utils.config import Config, ConfigError


class RothermelSimulationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config("./src/utils/_tests/test_configs/test_config.yml")
        self.config_flat_simple = Config(
            Path(self.config.path).parent / "test_config_flat_simple.yml"
        )

        self.screen_size = (self.config.area.screen_size, self.config.area.screen_size)

        self.simulation = RothermelSimulation(self.config)
        self.simulation_flat = RothermelSimulation(self.config_flat_simple)

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
        Test that the call to `get_actions()` runs properly and returns all Rothermel
        `FireLineManager()` features.
        """
        simulation_actions = self.simulation.get_actions()
        self.assertIsInstance(simulation_actions, Dict)

    def test_get_attribute_bounds(self) -> None:
        """
        Test that the call to get_actions() runs properly and returns all Rothermel
        features (Fire, Wind, FireLine, Terrain).
        """
        simulation_attributes = self.simulation.get_attribute_bounds()
        self.assertIsInstance(simulation_attributes, Dict)

    def test_get_attribute_data(self) -> None:
        """
        Test that the call to get_actions() runs properly and returns all Rothermel
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

        self.fire_map = self.simulation_flat.run(time="1h")
        # assert the fire map is all BURNED
        self.assertEqual(
            self.fire_map.max(),
            fire_map.max(),
            msg=f"The fire map has a maximum BurnStatus of {self.fire_map.max()} "
            f", but it should be {fire_map.max()}",
        )

        self.simulation_flat.reset()

        # Check that we can run for one step
        self.fire_map = self.simulation_flat.run(time=1)
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
        seeds = {"elevation": seed, "fuel": seed, "wind_speed": seed}
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
        self.assertDictEqual(
            seeds,
            returned_seeds,
            msg=f"The input seeds ({seeds}) do not match the returned seeds "
            f"({returned_seeds})",
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
        old_map = np.copy(self.simulation.fire_map)

        new_map = np.zeros((9, 9))
        new_map[0][0] = 10

        self.assertWarns(Warning, self.simulation.load_mitigation, new_map)
        self.assertTrue(np.array_equal(old_map, self.simulation.fire_map))

        new_map[0][0] = 3
        self.simulation.load_mitigation(new_map)

        self.assertTrue(np.array_equal(new_map, self.simulation.fire_map))
