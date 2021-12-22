import unittest

import numpy as np
import pygame

from ..sprites import Fire, Terrain
from ...utils.config import Config
from ...enums import BURNED_RGB_COLOR, BurnStatus, SpriteLayer


class TestTerrain(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config('./config.yml')
        self.tiles = [[
            self.config.terrain.fuel_array_function(x, y)
            for x in range(self.config.area.terrain_size)
        ] for y in range(self.config.area.terrain_size)]

        self.terrain = Terrain(self.tiles, self.config.terrain.elevation_function,
                               self.config.area.terrain_size,
                               self.config.area.screen_size)

    def test_image_loc_and_size(self) -> None:
        '''
        Test that the terrain image location and size is correct.
        '''
        x, y, w, h = self.terrain.rect
        self.assertTupleEqual((x, y), (0, 0),
                              msg=('The terrain image/texture location is '
                                   f'({x}, {y}), but should be (0, 0)'))
        valid_tuple = (self.config.area.screen_size, self.config.area.screen_size)
        self.assertTupleEqual((w, h),
                              valid_tuple,
                              msg=('The terrain image/texture location is '
                                   f'({w}, {h}), but should be '
                                   f'{valid_tuple}'))

    def test_image_creation(self) -> None:
        '''
        Test that the image with contour map, FuelArray array, and elevation array are
        created successfully.
        '''
        self.terrain._make_terrain_image()
        true_shape = (self.config.area.screen_size, self.config.area.screen_size)
        self.assertIsInstance(self.terrain.image,
                              pygame.Surface,
                              msg=('The terrain image should have type pygame.Surface, '
                                   f'but has type {type(self.terrain.image)}'))
        self.assertCountEqual(self.terrain.fuel_arrs.shape,
                              true_shape,
                              msg=('The FuelArray image has shape '
                                   f'{self.terrain.fuel_arrs.shape}, but should have '
                                   f'shape {true_shape}'))
        self.assertCountEqual(self.terrain.elevations.shape,
                              true_shape,
                              msg=('The terrain elevations have shape '
                                   f'{self.terrain.elevations.shape}, but should have '
                                   f'shape {true_shape}'))

    def test_update(self) -> None:
        '''
        Test that the terrain updates the correct pixels/tiles to burned.
        '''
        _, _, w, h = self.terrain.rect

        fire_map = np.full((h, w), BurnStatus.UNBURNED)

        burned_x = self.config.area.screen_size // 2
        burned_y = self.config.area.screen_size // 2
        fire_map[burned_y, burned_x] = BurnStatus.BURNED

        self.terrain.update(fire_map)

        terrain_img = self.terrain.image
        terrain_arr = pygame.surfarray.pixels3d(terrain_img).copy()

        self.assertTupleEqual(tuple(terrain_arr[burned_y, burned_x].tolist()),
                              BURNED_RGB_COLOR,
                              msg='The terrain failed to update the correct pixel/tile')

    def test_layer(self) -> None:
        '''
        Test that the Terrain has the correct Sprite/render layer.
        '''
        self.assertEqual(self.terrain.layer,
                         SpriteLayer.TERRAIN,
                         msg=(f'The terrain should have layer={SpriteLayer.TERRAIN}, '
                              f'but has layer={self.terrain.layer}'))


class TestFire(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config('./config.yml')
        self.pos = (self.config.area.screen_size // 2, self.config.area.screen_size // 2)
        self.size = self.config.display.fire_size
        self.fire = Fire(self.pos, self.size)

    def test_image_loc_and_size(self) -> None:
        '''
        Test that the terrain image location and size is correct.
        '''
        x, y, w, h = self.fire.rect

        self.assertTupleEqual((x, y),
                              self.pos,
                              msg=(f'The fire has (x, y) location of ({x}, {y}), '
                                   f'but should have location of {self.pos}'))

        self.assertTupleEqual((w, h), (self.size, self.size),
                              msg=(f'The fire has (width, height) size of ({w}, {h}), '
                                   f'but should have size of ({self.size}, {self.size})'))

    def test_layer(self) -> None:
        '''
        Test that the Fire has the correct Sprite/render layer.
        '''
        self.assertEqual(self.fire.layer,
                         SpriteLayer.FIRE,
                         msg=(f'The terrain should have layer={SpriteLayer.FIRE}, '
                              f'but has layer={self.fire.layer}'))
