from typing import Sequence

import numpy as np
import pygame

from .image import load_image
from .sprites import Fire


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

    def update(self, terrain, fire_sprites: Sequence[Fire], fire_map: np.ndarray) -> bool:
        '''
        Update the game display using the provided terrain, sprites, and
        environment data. Most of the logic for the game is handled within
        each sprite/manager, so this is mostly just calls to update everything.

        Arguments:
            terrain: The Terrain class that comprises the burnable area.
            fire_sprites: A list of all Fire sprites that are actively burning.
            fire_map: The screen_size x screen_size array that monitors each
                      pixel's burn status.
        '''
        running = True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Create a layered group so that the fire appears on top
        fire_sprites_group = pygame.sprite.LayeredUpdates(fire_sprites)
        all_sprites = pygame.sprite.LayeredUpdates(fire_sprites_group, terrain)

        # Update and draw the sprites
        for sprite in all_sprites.sprites():
            self.screen.blit(self.background, sprite.rect, sprite.rect)

        fire_sprites_group.update()
        terrain.update(fire_map)
        all_sprites.draw(self.screen)
        pygame.display.flip()

        return running
