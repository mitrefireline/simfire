from typing import Tuple

import numpy as np
import pygame

import src.config as cfg
from src.game.Game import Game
from src.game.managers import FireManager
from src.game.sprites import GameTile
from src.world.parameters import FuelArray, Tile


def main():
    pygame.init()
    game = Game(cfg.screen_size)

    game_tiles = make_game_tiles()
    game_tiles_sprites = pygame.sprite.RenderPlain(game_tiles.tolist())

    fire_manager = FireManager(cfg.fire_init_pos, cfg.fire_size,
                               cfg.max_fire_duration, cfg.rate_of_spread)

    running = True
    while running:
        fire_sprites = fire_manager.sprites
        fire_map = fire_manager.fire_map
        running = game.update(game_tiles.tolist(), fire_sprites, fire_map)
        fire_manager.update()

    pygame.quit()


def make_game_tiles() -> np.ndarray:
    '''
    Create numpy array of shape (cfg.terrain_size, cfg.terrain_size) containing
    a GameTile for each entry.

    Arguments:
        None

    Returns:
        game_tiles: Numpy array containing the GameTiles
    '''
    fuel_arrs = [[FuelArray(Tile(j, i, 0),
                            cfg.w_0[j, i],
                            cfg.delta[j, i],
                            cfg.M_x[j, i]) \
                  for j in range(cfg.terrain_size)] \
                  for i in range(cfg.terrain_size)]
    fuel_arrs = np.array(fuel_arrs)

    game_tile_size = cfg.screen_size // cfg.terrain_size
    game_tiles = [[GameTile(fuel_arrs[j, i], game_tile_size, (j, i)) \
                   for j in range(cfg.terrain_size)] \
                   for i in range(cfg.terrain_size)]
    game_tiles = np.array(game_tiles)
    return game_tiles


if __name__ == '__main__':
    main()