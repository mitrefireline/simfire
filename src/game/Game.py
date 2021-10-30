from typing import Sequence

import numpy as np
import pygame
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
        pygame.display.set_icon(load_image('assets/icons/fireline_logo.png'))

        # Create the background so it doesn't have to be recreated every update
        self.background = pygame.Surface(self.screen.get_size())
        self.background = self.background.convert()
        self.background.fill((0, 0, 0))
        # Map to track which pixels are on fire or have burned
        self.fire_map = np.full(pygame.display.get_surface().get_size(),
                                BurnStatus.UNBURNED)

        self.show_wind_magnitude = False
        self.show_wind_direction = False

    def _toggle_wind_magnitude_display(self, switch: bool):
        '''
        Toggle display of wind MAGNITUDE over the main screen
        Arguments: None
        '''
        self.show_wind_magnitude = switch
        if self.show_wind_magnitude is False:
            print('Wind Magnitude OFF')
        else:
            print('Wind Magnitude ON')
        return

    def _toggle_wind_direction_display(self, switch: bool):
        '''
        Toggle display of wind DIRECTION over the main screen
        Arguments: None
        '''
        self.show_wind_direction = switch
        if self.show_wind_direction is False:
            print('Wind Direction OFF')
        else:
            print('Wind Direction ON')
        return

    def update(self, terrain: Terrain, fire_sprites: Sequence[Fire],
               fireline_sprites: Sequence[FireLine],
               wind_magnitude_map: Sequence[Sequence[float]]) -> bool:
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
                    self._toggle_wind_magnitude_display(True)

                if keys[pygame.K_n] is True:
                    self._toggle_wind_direction_display(True)

                if keys[pygame.K_k] is True:
                    self._toggle_wind_magnitude_display(False)

                if keys[pygame.K_j] is True:
                    self._toggle_wind_direction_display(False)

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
        pygame.display.flip()

        return status
