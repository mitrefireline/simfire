from typing import Tuple

import numpy as np
import pygame

import src.config as cfg
from src.game.Game import Game
from src.game.managers import ConstantSpreadFireManager
from src.game.sprites import Terrain
from src.world.parameters import FuelArray, Tile


def main():
    pygame.init()
    game = Game(cfg.screen_size)

    fuel_arrs = [[FuelArray(Tile(j, i, 0),
                            cfg.fuel.w_0[j, i],
                            cfg.fuel.delta[j, i],
                            cfg.fuel.M_x[j, i]) \
                  for j in range(cfg.terrain_size)] \
                  for i in range(cfg.terrain_size)]
    terrain = Terrain(fuel_arrs)

    # The number of frames it takes the fire to spread 1 pixel
    rate_of_spread = 5
    fire_manager = ConstantSpreadFireManager(cfg.fire_init_pos, cfg.fire_size,
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