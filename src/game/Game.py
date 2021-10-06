from typing import Sequence

import numpy as np
import pygame

from .image import load_image
from .sprites import Fire, GameTile


class Game():
    def __init__(self, screen_size: int) -> None:
        self.screen_size = 900
        self.screen = pygame.display.set_mode((screen_size, screen_size))
        pygame.display.set_caption('Rothermel 2D Simulator')
        pygame.display.set_icon(load_image('assets/icons/fireline_logo.png'))
        self.background = pygame.Surface(self.screen.get_size())
        self.background = self.background.convert()
        self.background.fill((0, 0, 0))

    def update(self,
               game_tiles_sprites: Sequence[GameTile],
               fire_sprites: Sequence[Fire],
               fire_map: np.ndarray) -> bool:
        running = True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Create a layered group so that the fire appears on top
        fire_sprites_group = pygame.sprite.LayeredUpdates(fire_sprites)
        tile_sprites_group = pygame.sprite.LayeredUpdates(game_tiles_sprites)
        all_sprites = pygame.sprite.LayeredUpdates(fire_sprites_group, tile_sprites_group)
        # all_sprites.add(game_tiles_sprites)
        # all_sprites.add(fire_sprites)

        # Update and draw the sprites
        for sprite in all_sprites.sprites():
            self.screen.blit(self.background, sprite.rect, sprite.rect)
        fire_sprites_group.update()
        tile_sprites_group.update(fire_map)
        all_sprites.draw(self.screen)
        pygame.display.flip()

        return running