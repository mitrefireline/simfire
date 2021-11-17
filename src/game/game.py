from typing import Sequence
from importlib import resources

import numpy as np
import pygame
import math
import src.config as cfg

from .image import load_image
from ..enums import BurnStatus
from .sprites import Fire, FireLine, Terrain
from ..enums import GameStatus


class Game():
    '''
    Class that controls the game. This class will initalize the game
    and allow for terrain, fire, and other sprites to be rendered and
    interact.
    '''
    def __init__(self, screen_size: int) -> None:
        '''
        Initalize the class by creating the game display and background.

        Arguments:
            screen_size: The (n,n) size of the game screen/display
        '''
        self.screen_size = screen_size
        self.screen = pygame.display.set_mode((screen_size, screen_size))

        pygame.display.set_caption('Rothermel 2D Simulator')
        with resources.path('assets.icons', 'fireline_logo.png') as path:
            fireline_logo_path = path
        pygame.display.set_icon(load_image(fireline_logo_path))

        # Create the background so it doesn't have to be recreated every update
        self.background = pygame.Surface(self.screen.get_size())
        self.background = self.background.convert()
        self.background.fill((0, 0, 0))
        # Map to track which pixels are on fire or have burned
        self.fire_map = np.full(pygame.display.get_surface().get_size(),
                                BurnStatus.UNBURNED)

        self.show_wind_magnitude = False
        self.show_wind_direction = False

    def _toggle_wind_magnitude_display(self):
        '''
        Toggle display of wind MAGNITUDE over the main screen
        Arguments: None
        '''
        self.show_wind_magnitude = not self.show_wind_magnitude
        if self.show_wind_magnitude is False:
            print('Wind Magnitude OFF')
        else:
            print('Wind Magnitude ON')
        return

    def _toggle_wind_direction_display(self):
        '''
        Toggle display of wind DIRECTION over the main screen
        Arguments: None
        '''
        self.show_wind_direction = not self.show_wind_direction
        if self.show_wind_direction is False:
            print('Wind Direction OFF')
        else:
            print('Wind Direction ON')
        return

    def _disable_wind_magnitude_display(self):
        '''
        Toggle display of wind DIRECTION over the main screen
        Arguments: None
        '''
        self.show_wind_magnitude = False

    def _disable_wind_direction_display(self):
        '''
        Toggle display of wind DIRECTION over the main screen
        Arguments: None
        '''
        self.show_wind_direction = False

    def _get_wind_direction_color(self, direction: float, ws_min: float,
                                  ws_max: float) -> (int, int, int):
        '''
        Get the color and intensity representing direction based
        on wind direction
        0/360: Black, 90: Red, 180: White, Blue: 270

        Returns tuple of RGBa where a is alpha channel or
        transparency of the color

        Arguments:
            direction: Float value of the angle 0-360
            ws_min: minimum wind speed
            ws_max: maximum wind speed
        '''
        north_min = 0.0
        north_max = 360.0
        east = 90.0
        south = 180.0
        west = 270.0

        colorRGB = (255.0, 255.0, 255.0)  # Default white

        # North to East, Red to Green
        if direction >= north_min and direction < east:
            angleRange = (east - north_min)

            # Red
            redMin = 255.0
            redMax = 128.0
            redRange = (redMax - redMin)  # 255 - 128 red range from north to east
            red = (((direction - north_min) * redRange) / angleRange) + redMin

            # Green
            greenMin = 0.0
            greenMax = 255.0
            greenRange = (greenMax - greenMin)  # 0 - 255 red range from north to east
            green = (((direction - north_min) * greenRange) / angleRange) + greenMin

            colorRGB = (red, green, 0.0)

        # East to South, Green to Teal
        if direction >= east and direction < south:
            angleRange = (south - east)

            # Red
            redMin = 128.0
            redMax = 0.0
            redRange = (redMax - redMin)  # 128 - 0 red range from east to south
            red = (((direction - east) * redRange) / angleRange) + redMin

            # Blue
            blueMin = 0
            blueMax = 255
            blueRange = (blueMax - blueMin)  # 0 - 255 blue range from east to south
            blue = (((direction - east) * blueRange) / angleRange) + blueMin

            colorRGB = (red, 255, blue)

        # South to West, Teal to Purple
        if direction >= south and direction < west:
            angleRange = (west - south)

            # Red
            redMin = 0
            redMax = 128
            redRange = (redMax - redMin)  # 0 - 128 red range from south to west
            red = (((direction - south) * redRange) / angleRange) + redMin

            # Green
            greenMin = 255
            greenMax = 0
            greenRange = (greenMax - greenMin)  # 0 - 255 green range from south to west
            green = (((direction - south) * greenRange) / angleRange) + greenMin

            colorRGB = (red, green, 255)

        # West to North, Purple to Red
        if direction <= north_max and direction >= west:
            angleRange = (north_max - west)

            # Red
            redMin = 128.0
            redMax = 255.0
            redRange = (redMax - redMin)  # 128 - 255 red range from east to south
            red = (((direction - west) * redRange) / angleRange) + redMin

            # Blue
            blueMin = 0
            blueMax = 255
            blueRange = (blueMax - blueMin)  # 0 - 255 blue range from east to south
            blue = (((direction - west) * blueRange) / angleRange) + blueMin

            colorRGB = (red, 0, blue)

        floorColorRGB = (
            int(math.floor(colorRGB[0])),
            int(math.floor(colorRGB[1])),
            int(math.floor(colorRGB[2])),
        )
        return floorColorRGB

    def update(self, terrain: Terrain, fire_sprites: Sequence[Fire],
               fireline_sprites: Sequence[FireLine],
               wind_magnitude_map: Sequence[Sequence[float]],
               wind_direction_map: Sequence[Sequence[float]]) -> bool:
        '''
        Update the game display using the provided terrain, sprites, and
        environment data. Most of the logic for the game is handled within
        each sprite/manager, so this is mostly just calls to update everything.

        Arguments:
            terrain: The Terrain class that comprises the burnable area.
            fire_sprites: A list of all Fire sprites that are actively burning.
            fireline_sprites: A list of all FireLine sprites that are dug.
        '''
        status = GameStatus.RUNNING

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                status = GameStatus.QUIT

            if event.type == pygame.KEYDOWN:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_m] is True:
                    self._disable_wind_direction_display()
                    self._toggle_wind_magnitude_display()

                if keys[pygame.K_n] is True:
                    self._disable_wind_magnitude_display()
                    self._toggle_wind_direction_display()

        # Create a layered group so that the fire appears on top
        fire_sprites_group = pygame.sprite.LayeredUpdates(fire_sprites, fireline_sprites)
        all_sprites = pygame.sprite.LayeredUpdates(fire_sprites_group, terrain)

        # if self.show_wind_magnitude is True:

        # if self.show_wind_direction is true

        # Update and draw the sprites
        for sprite in all_sprites.sprites():
            self.screen.blit(self.background, sprite.rect, sprite.rect)

        fire_sprites_group.update()
        terrain.update(self.fire_map)
        all_sprites.draw(self.screen)

        if self.show_wind_magnitude is True:
            wind_mag_surf = pygame.Surface(self.screen.get_size())
            for y_idx, y in enumerate(wind_magnitude_map):
                for x_idx, x in enumerate(y):
                    w_mag = x
                    wind_speed_range = (cfg.mw_speed_max - cfg.mw_speed_min)
                    color_grad = (255 - 0)
                    color_mag = int(((
                        (w_mag - cfg.mw_speed_min) * color_grad) / wind_speed_range) + 0)
                    wind_mag_surf.set_at((x_idx, y_idx), pygame.Color(0,
                                                                      0,
                                                                      color_mag,
                                                                      a=1))
            self.screen.blit(wind_mag_surf, (0, 0))

        if self.show_wind_direction is True:
            wind_dir_surf = pygame.Surface(self.screen.get_size())
            for y_idx, y in enumerate(wind_direction_map):
                for x_idx, x in enumerate(y):
                    w_dir = x
                    color = self._get_wind_direction_color(w_dir, cfg.dw_deg_min,
                                                           cfg.dw_deg_max)
                    pyColor = pygame.Color(color[0], color[1], color[2], a=0.75)
                    wind_dir_surf.set_at((x_idx, y_idx), pyColor)
            self.screen.blit(wind_dir_surf, (0, 0))
        pygame.display.flip()

        return status
