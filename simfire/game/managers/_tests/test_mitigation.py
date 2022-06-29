import os
import unittest

import numpy as np
from skimage.draw import line

from ....enums import BurnStatus
from ....game._tests import DummyFuelLayer, DummyTopographyLayer
from ....game.game import Game
from ....utils.config import Config
from ...sprites import FireLine, ScratchLine, Terrain, WetLine
from ..mitigation import FireLineManager, ScratchLineManager, WetLineManager

# unittest.mock.patch.dict isn't working anymore
# Can't run rests without setting display at the top of the file
os.environ["SDL_VIDEODRIVER"] = "dummy"


class TestControlLineManager(unittest.TestCase):
    """
    Tests the parent `ControlLineManager` class's `add_point` and `update` functions
    using the `FireLineManager` as a proxy.
    """

    def setUp(self) -> None:
        self.config = Config("./simfire/utils/_tests/test_configs/test_config.yml")
        self.screen_size = (self.config.area.screen_size, self.config.area.screen_size)
        points = line(self.screen_size[0] // 4, 0, 0, self.screen_size[1] // 4)
        y = points[0].tolist()
        x = points[1].tolist()
        self.manager = FireLineManager
        self.points = list(zip(x, y))
        self.game = Game(self.screen_size)
        topo_layer = DummyTopographyLayer(self.screen_size)
        fuel_layer = DummyFuelLayer(self.screen_size)
        self.terrain = Terrain(fuel_layer, topo_layer, self.screen_size)

    def test_add_point(self) -> None:
        """
        Test that a point is added to the self.sprites list correctly.
        """
        manager = self.manager(
            size=self.config.display.control_line_size,
            pixel_scale=self.config.area.pixel_scale,
            terrain=self.terrain,
        )

        manager._add_point(self.points[0])

        self.assertEqual(
            len(manager.sprites),
            1,
            msg=("The ControlLineManager did not add a point to its " "self.sprites."),
        )

        self.assertIsInstance(
            manager.sprites[0],
            manager.sprite_type,
            msg=(
                "The ControlLineManager did not add the correct "
                "sprite type to self.sprites in self._add_point."
            ),
        )

    def test_update(self) -> None:
        """
        Test that all points supplied to `manager.update()` are the same points in the
        resulting `fire_map`
        """
        fire_map = np.full(
            (self.config.area.screen_size, self.config.area.screen_size),
            BurnStatus.UNBURNED,
        )

        manager = self.manager(
            size=self.config.display.control_line_size,
            pixel_scale=self.config.area.pixel_scale,
            terrain=self.terrain,
        )
        fire_map = self.game.fire_map
        fire_map = manager.update(fire_map, self.points)

        fireline_points = np.argwhere(fire_map == BurnStatus.FIRELINE)

        # self.points is (x, y) whereas fire_map is (y, x)
        fireline_points = [(x, y) for y, x in fireline_points]

        # Test to see if we have the same locations in the resulting fire_map that we
        # supplied to self.update
        self.assertCountEqual(
            fireline_points,
            self.points,
            msg=(
                "The points given as parameters to "
                f"FireLineManager.update(): {self.points} do not "
                "match those in the returned fire_map: "
                f"{fireline_points}"
            ),
        )


class TestFireLineManager(unittest.TestCase):
    """
    Meant to test the `FireLineManager` class. Will eventually test the different physics
    present in FireLines, but right now assumes that they stop the fire in its tracks.
    """

    def setUp(self) -> None:
        self.config = Config("./simfire/utils/_tests/test_configs/test_config.yml")
        self.screen_size = (self.config.area.screen_size, self.config.area.screen_size)
        points = line(self.screen_size[0] // 4, 0, 0, self.screen_size[1] // 4)
        y = points[0].tolist()
        x = points[1].tolist()
        self.manager = FireLineManager
        self.points = list(zip(x, y))
        self.game = Game(self.screen_size)
        topo_layer = DummyTopographyLayer(self.screen_size)
        fuel_layer = DummyFuelLayer(self.screen_size)
        self.terrain = Terrain(fuel_layer, topo_layer, self.screen_size)

    def test_init(self) -> None:
        """
        Test to make sure that `self.sprite_type` and `self.line_type` are set correctly
        """
        manager = self.manager(
            size=self.config.display.control_line_size,
            pixel_scale=self.config.area.pixel_scale,
            terrain=self.terrain,
        )

        # Make sure line_type is the same
        self.assertEqual(
            manager.line_type,
            BurnStatus.FIRELINE,
            msg=("FireLine.line_type is different than " "BurnStatus.FIRELINE"),
        )

        # Make sure sprite_type is the same
        self.assertEqual(
            manager.sprite_type,
            FireLine,
            msg=("FireLine.sprite_type is not a FireLine sprite"),
        )


class TestScratchLineManager(unittest.TestCase):
    """
    Meant to test the `ScratchLineManager` class. Will eventually test the different
    physics present in ScratchLines.
    """

    def setUp(self) -> None:
        self.config = Config("./simfire/utils/_tests/test_configs/test_config.yml")
        self.screen_size = (self.config.area.screen_size, self.config.area.screen_size)
        points = line(self.screen_size[0] // 4, 0, 0, self.screen_size[1] // 4)
        y = points[0].tolist()
        x = points[1].tolist()
        self.manager = ScratchLineManager
        self.points = list(zip(x, y))
        self.game = Game(self.screen_size)
        topo_layer = DummyTopographyLayer(self.screen_size)
        fuel_layer = DummyFuelLayer(self.screen_size)
        self.terrain = Terrain(fuel_layer, topo_layer, self.screen_size)

    def test_init(self) -> None:
        """
        Test to make sure that `self.sprite_type` and `self.line_type` are set correctly
        """
        manager = self.manager(
            size=self.config.display.control_line_size,
            pixel_scale=self.config.area.pixel_scale,
            terrain=self.terrain,
        )

        # Make sure line_type is the same
        self.assertEqual(
            manager.line_type,
            BurnStatus.SCRATCHLINE,
            msg=("FireLine.line_type is different than " "BurnStatus.FIRELINE"),
        )

        # Make sure sprite_type is the same
        self.assertEqual(
            manager.sprite_type,
            ScratchLine,
            msg=("FireLine.sprite_type is not a FireLine sprite"),
        )


class TestWetLineManager(unittest.TestCase):
    """
    Meant to test the `WetLineManager` class. Will eventually test the different physics
    present in WetLines.
    """

    def setUp(self) -> None:
        self.config = Config("./simfire/utils/_tests/test_configs/test_config.yml")
        self.screen_size = (self.config.area.screen_size, self.config.area.screen_size)
        points = line(self.screen_size[0] // 4, 0, 0, self.screen_size[1] // 4)
        y = points[0].tolist()
        x = points[1].tolist()
        self.manager = WetLineManager
        self.points = list(zip(x, y))
        self.game = Game(self.screen_size)
        topo_layer = DummyTopographyLayer(self.screen_size)
        fuel_layer = DummyFuelLayer(self.screen_size)
        self.terrain = Terrain(fuel_layer, topo_layer, self.screen_size)

    def test_init(self) -> None:
        """
        Test to make sure that `self.sprite_type` and `self.line_type` are set correctly
        """
        manager = self.manager(
            size=self.config.display.control_line_size,
            pixel_scale=self.config.area.pixel_scale,
            terrain=self.terrain,
        )

        # Make sure line_type is the same
        self.assertEqual(
            manager.line_type,
            BurnStatus.WETLINE,
            msg=("FireLine.line_type is different than " "BurnStatus.FIRELINE"),
        )

        # Make sure sprite_type is the same
        self.assertEqual(
            manager.sprite_type,
            WetLine,
            msg=("FireLine.sprite_type is not a FireLine sprite"),
        )
