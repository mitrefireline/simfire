import unittest

import numpy as np
import pygame

from ..sprites import Fire, Terrain
from ... import config as cfg
from ...enums import BURNED_RGB_COLOR, BurnStatus, SpriteLayer
from ...world.parameters import FuelArray, Tile


class TestTerrain(unittest.TestCase):
    def setUp(self) -> None:
        self.tiles = [[
            FuelArray(Tile(j, i, cfg.terrain_scale, cfg.terrain_scale),
                      cfg.terrain_map[i][j]) for j in range(cfg.terrain_size)
        ] for i in range(cfg.terrain_size)]

        self.terrain = Terrain(self.tiles, cfg.elevation_fn)

    def test_image_loc_and_size(self) -> None:
        '''
        Test that the terrain image location and size is correct.
        '''
        x, y, w, h = self.terrain.rect
        self.assertTupleEqual((x, y), (0, 0),
                              msg=('The terrain image/texture location is '
                                   f'({x}, {y}), but should be (0, 0)'))
        self.assertTupleEqual((w, h), (cfg.screen_size, cfg.screen_size),
                              msg=('The terrain image/texture location is '
                                   f'({w}, {h}), but should be '
                                   f'({cfg.screen_size}, {cfg.screen_size})'))

    def test_image_creation(self) -> None:
        '''
        Test that the image with contour map, FuelArray array, and elevation array are
        created successfully.
        '''
        self.terrain._make_terrain_image()
        true_shape = (cfg.screen_size, cfg.screen_size)
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

        burned_x = cfg.screen_size // 2
        burned_y = cfg.screen_size // 2
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
        self.pos = (cfg.screen_size // 2, cfg.screen_size // 2)
        self.size = cfg.fire_size
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

    def test_update(self) -> None:
        '''
        Test that the fire updates the duration counter correctly.
        '''
        num_updates = 5
        for _ in range(num_updates):
            self.fire.update()

        self.assertEqual(self.fire.duration,
                         num_updates,
                         msg=(f'The fire duration is {self.fire.duration} frames, '
                              f'but should be {num_updates} frames'))

    def test_layer(self) -> None:
        '''
        Test that the Fire has the correct Sprite/render layer.
        '''
        self.assertEqual(self.fire.layer,
                         SpriteLayer.FIRE,
                         msg=(f'The terrain should have layer={SpriteLayer.FIRE}, '
                              f'but has layer={self.fire.layer}'))
