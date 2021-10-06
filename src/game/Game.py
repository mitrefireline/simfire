import numpy as np
import pygame

from .image import load_image


class Game():
    def __init__(self, screen_size: int) -> None:
        self.screen_size = 900
        self.screen = pygame.display.set_mode((screen_size, screen_size))
        pygame.display.set_caption('Rothermel 2D Simulator')
        self.background = pygame.Surface(self.screen.get_size())
        self.background = self.background.convert()
        self.background.fill((0, 0, 0))

    def update(self,
               game_tiles_sprites: pygame.sprite.Group,
               fire_sprites: pygame.sprite.Group,
               ) -> bool:
        running = True

        all_sprites = pygame.sprite.LayeredUpdates()
        all_sprites.add(game_tiles_sprites.sprites())
        all_sprites.add(fire_sprites.sprites())

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        for sprite in all_sprites.sprites():
            self.screen.blit(self.background, sprite.rect, sprite.rect)
        all_sprites.update()
        all_sprites.draw(self.screen)
        # for sprite in fire_sprites.sprites():
        #     self.screen.blit(self.background, sprite.rect, sprite.rect)
        # for sprite in game_tiles_sprites.sprites():
        #     self.screen.blit(self.background, sprite.rect, sprite.rect)

        # fire_sprites.update()
        # fire_sprites.draw(self.screen)
        # game_tiles_sprites.update()
        # game_tiles_sprites.draw(self.screen)
        pygame.display.flip()

        return running