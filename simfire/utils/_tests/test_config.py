import unittest
from pathlib import Path

import yaml

from ..config import Config


class ConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.yaml = Path("./simfire/utils/_tests/test_configs/test_config.yml")
        self.cfg = Config(self.yaml)
        with open(self.yaml, "r") as f:
            self.true_yaml_data = yaml.safe_load(f)

    def test_reset_terrain(self) -> None:
        """
        Test resetting the seed and coordinates for the terrain topography
        and fuel layers.
        """
        # test_config.yml has a seed of 1111
        new_topo_seed = 1234
        new_fuel_seed = 4321
        new_loc = (39.65, 119.75)
        self.cfg.reset_terrain(
            topography_seed=new_topo_seed, fuel_seed=new_fuel_seed, location=new_loc
        )

        # The seeds should be updated after calling the reset method
        yaml_data = self.cfg.yaml_data
        topo_fn_name = yaml_data["terrain"]["topography"]["functional"]["function"]
        cfg_seed = yaml_data["terrain"]["topography"]["functional"][topo_fn_name]["seed"]
        self.assertEqual(
            new_topo_seed,
            cfg_seed,
            msg=f"The assigned seed of {cfg_seed} does "
            f"not match the test seed of {new_topo_seed}",
        )

        fuel_fn_name = yaml_data["terrain"]["fuel"]["functional"]["function"]
        cfg_seed = yaml_data["terrain"]["fuel"]["functional"][fuel_fn_name]["seed"]
        self.assertEqual(
            new_fuel_seed,
            cfg_seed,
            msg=f"The assigned seed of {cfg_seed} does "
            f"not match the test seed of {new_topo_seed}",
        )

    def test_reset_wind_function(self) -> None:
        """
        Test resetting the seed for the wind function and returning a different map
        """
        pass

    def test_save(self) -> None:
        """
        Test saving the config's data and making sure it matches the original YAML
        """
        save_path = self.yaml.parent / "save_config.yml"
        self.cfg.save(save_path)
        with open(save_path, "r") as f:
            save_data = yaml.safe_load(f)
        save_path.unlink()
        self.assertDictEqual(
            self.cfg.yaml_data,
            save_data,
            msg=f"The data in the saved YAML at {save_path} does not "
            f"match the data in the test YAML at {self.yaml}",
        )
