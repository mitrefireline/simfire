import unittest

import numpy as np
import pygame

from ...enums import BURNED_RGB_COLOR, BurnStatus, SpriteLayer
from ...utils.config import Config
from ..sprites import Fire, FireLine, ScratchLine, Terrain, WetLine
from . import DummyFuelLayer, DummyTopographyLayer


class TestTerrain(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config("./simfire/utils/_tests/test_configs/test_config.yml")
        self.screen_size = (32, 32)
        self.fuel_layer = DummyFuelLayer(self.screen_size)
        self.topo_layer = DummyTopographyLayer(self.screen_size)
        self.terrain = Terrain(self.fuel_layer, self.topo_layer, self.screen_size)

    def test_image_loc_and_size(self) -> None:
        """
        Test that the terrain image location and size is correct.
        """
        x, y, w, h = self.terrain.rect
        self.assertTupleEqual(
            (x, y),
            (0, 0),
            msg=(
                "The terrain image/texture location is "
                f"({x}, {y}), but should be (0, 0)"
            ),
        )
        self.assertTupleEqual(
            (w, h),
            self.screen_size,
            msg=(
                "The terrain image/texture location is "
                f"({w}, {h}), but should be "
                f"{self.screen_size}"
            ),
        )

    def test_image_creation(self) -> None:
        """
        Test that the image with contour map, Fuel array, and elevation array are
        created successfully.
        """
        self.terrain._make_terrain_image()
        self.assertIsInstance(
            self.terrain.image,
            pygame.Surface,
            msg=(
                "The terrain image should have type pygame.Surface, "
                f"but has type {type(self.terrain.image)}"
            ),
        )
        self.assertCountEqual(
            self.terrain.fuels.shape,
            self.screen_size,
            msg=(
                "The terrain fuels have shape "
                f"{self.terrain.fuels.shape}, but should have "
                f"shape {self.screen_size}"
            ),
        )
        self.assertCountEqual(
            self.terrain.elevations.shape,
            self.screen_size,
            msg=(
                "The terrain elevations have shape "
                f"{self.terrain.elevations.shape}, but should have "
                f"shape {self.screen_size}"
            ),
        )

    def test_update(self) -> None:
        """
        Test that the terrain updates the correct pixels/tiles to burned.
        """
        _, _, w, h = self.terrain.rect

        fire_map = np.full((h, w), BurnStatus.UNBURNED)

        burned_x = self.screen_size[1] // 2
        burned_y = self.screen_size[0] // 2
        fire_map[burned_y, burned_x] = BurnStatus.BURNED

        self.terrain.update(fire_map)

        terrain_img = self.terrain.image
        terrain_arr = pygame.surfarray.pixels3d(terrain_img).copy()

        self.assertTupleEqual(
            tuple(terrain_arr[burned_y, burned_x].tolist()),
            BURNED_RGB_COLOR,
            msg="The terrain failed to update the correct pixel/tile",
        )

    def test_layer(self) -> None:
        """
        Test that the Terrain has the correct Sprite/render layer.
        """
        self.assertEqual(
            self.terrain.layer,
            SpriteLayer.TERRAIN,
            msg=(
                f"The terrain should have layer={SpriteLayer.TERRAIN}, "
                f"but has layer={self.terrain.layer}"
            ),
        )


class TestFire(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config("./simfire/utils/_tests/test_configs/test_config.yml")
        self.pos = (self.config.area.screen_size // 2, self.config.area.screen_size // 2)
        self.size = self.config.display.fire_size
        self.fire = Fire(self.pos, self.size)

    def test_image_loc_and_size(self) -> None:
        """
        Test that the terrain image location and size is correct.
        """
        x, y, w, h = self.fire.rect

        self.assertTupleEqual(
            (x, y),
            self.pos,
            msg=(
                f"The fire has (x, y) location of ({x}, {y}), "
                f"but should have location of {self.pos}"
            ),
        )

        self.assertTupleEqual(
            (w, h),
            (self.size, self.size),
            msg=(
                f"The fire has (width, height) size of ({w}, {h}), "
                f"but should have size of ({self.size}, {self.size})"
            ),
        )

    def test_layer(self) -> None:
        """
        Test that the Fire has the correct Sprite/render layer.
        """
        self.assertEqual(
            self.fire.layer,
            SpriteLayer.FIRE,
            msg=(
                f"The terrain should have layer={SpriteLayer.FIRE}, "
                f"but has layer={self.fire.layer}"
            ),
        )


class TestFireLine(unittest.TestCase):
    def setUp(self) -> None:
        self.pos = (0, 0)
        self.size = 1
        return super().setUp()

    def test_headless(self) -> None:
        """Test that the FireLine sprite runs in a headless state"""
        headless = True
        sprite = FireLine(self.pos, self.size, headless)
        self.assertIsNone(
            sprite.image, msg="The sprite image should be None when headless==True"
        )

    def test_non_headless(self) -> None:
        """Test that the FireLine sprite runs in a non-headless state"""
        headless = False
        sprite = FireLine(self.pos, self.size, headless)
        self.assertIsInstance(
            sprite.image,
            pygame.Surface,
            msg="The sprite image should a pygame.Surface, but got"
            f"{type(sprite.image)}",
        )

        valid_size = (self.size, self.size)
        self.assertEqual(
            sprite.image.get_size(),
            valid_size,
            msg=f"The sprite image size should be {valid_size}, but got "
            f"{sprite.image.get_size()}",
        )

        rgba = sprite.image.get_at((0, 0))
        valid_rgba = (155, 118, 83, 255)
        self.assertEqual(
            rgba,
            valid_rgba,
            msg="The valid RGBA value for the sprite is {valid_rgba}, but " "got {rgba}",
        )

    def test_update(self) -> None:
        """Test that the FireLine.update() call returns None"""
        headless = True
        sprite = FireLine(self.pos, self.size, headless)
        self.assertIsNone(
            sprite.update(), msg="The sprite update step should return None"
        )


class TestScratchLine(unittest.TestCase):
    def setUp(self) -> None:
        self.pos = (0, 0)
        self.size = 1
        return super().setUp()

    def test_headless(self) -> None:
        """Test that the ScratchLine sprite runs in a headless state"""
        headless = True
        sprite = ScratchLine(self.pos, self.size, headless)
        self.assertIsNone(
            sprite.image, msg="The sprite image should be None when headless==True"
        )

    def test_non_headless(self) -> None:
        """Test that the ScratchLine sprite runs in a non-headless state"""
        headless = False
        sprite = ScratchLine(self.pos, self.size, headless)
        self.assertIsInstance(
            sprite.image,
            pygame.Surface,
            msg="The sprite image should a pygame.Surface, but got"
            f"{type(sprite.image)}",
        )

        valid_size = (self.size, self.size)
        self.assertEqual(
            sprite.image.get_size(),
            valid_size,
            msg=f"The sprite image size should be {valid_size}, but got "
            f"{sprite.image.get_size()}",
        )

        rgba = sprite.image.get_at((0, 0))
        valid_rgba = (139, 125, 58, 255)
        self.assertEqual(
            rgba,
            valid_rgba,
            msg="The valid RGBA value for the sprite is {valid_rgba}, but " "got {rgba}",
        )

    def test_update(self) -> None:
        """Test that the ScratchLine.update() call returns None"""
        headless = True
        sprite = FireLine(self.pos, self.size, headless)
        self.assertIsNone(
            sprite.update(), msg="The sprite update step should return None"
        )


class WetFireLine(unittest.TestCase):
    def setUp(self) -> None:
        self.pos = (0, 0)
        self.size = 1
        return super().setUp()

    def test_headless(self) -> None:
        """Test that the WetLine sprite runs in a headless state"""
        headless = True
        sprite = WetLine(self.pos, self.size, headless)
        self.assertIsNone(
            sprite.image, msg="The sprite image should be None when headless==True"
        )

    def test_non_headless(self) -> None:
        """Test that the WetLine sprite runs in a non-headless state"""
        headless = False
        sprite = WetLine(self.pos, self.size, headless)
        self.assertIsInstance(
            sprite.image,
            pygame.Surface,
            msg="The sprite image should a pygame.Surface, but got"
            f"{type(sprite.image)}",
        )

        valid_size = (self.size, self.size)
        self.assertEqual(
            sprite.image.get_size(),
            valid_size,
            msg=f"The sprite image size should be {valid_size}, but got "
            f"{sprite.image.get_size()}",
        )

        rgba = sprite.image.get_at((0, 0))
        valid_rgba = (212, 241, 249, 255)
        self.assertEqual(
            rgba,
            valid_rgba,
            msg="The valid RGBA value for the sprite is {valid_rgba}, but " "got {rgba}",
        )

    def test_update(self) -> None:
        """Test that the WetLine.update() call returns None"""
        headless = True
        sprite = WetLine(self.pos, self.size, headless)
        self.assertIsNone(
            sprite.update(), msg="The sprite update step should return None"
        )
