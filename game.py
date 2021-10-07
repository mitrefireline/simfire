from typing import Tuple

import numpy as np
import pygame

import src.config as cfg
from src.game.Game import Game
from src.game.managers import FireManager
from src.game.sprites import Terrain
from src.world.parameters import FuelArray, Tile


def main():
    pygame.init()
    game = Game(cfg.screen_size)

    fuel_arrs = [[FuelArray(Tile(j, i, 0),
                            cfg.w_0[j, i],
                            cfg.delta[j, i],
                            cfg.M_x[j, i]) \
                  for j in range(cfg.terrain_size)] \
                  for i in range(cfg.terrain_size)]
    terrain = Terrain(fuel_arrs)

    fire_manager = FireManager(cfg.fire_init_pos, cfg.fire_size,
                               cfg.max_fire_duration, cfg.rate_of_spread)

    running = True
    while running:
        fire_sprites = fire_manager.sprites
        fire_map = fire_manager.fire_map
        running = game.update(terrain, fire_sprites, fire_map)
        fire_manager.update()

    pygame.quit()


if __name__ == '__main__':
    main()