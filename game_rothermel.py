import pygame

import src.config as cfg
from src.game.Game import Game
from src.game.managers import RothermelFireManager
from src.game.sprites import Terrain
from src.world.parameters import Environment, FuelArray, FuelParticle, Tile


def main():
    pygame.init()
    game = Game(cfg.screen_size)

    fuel_particle = FuelParticle()

    fuel_arrs = [[
        FuelArray(Tile(j, i, 0, cfg.terrain_scale, cfg.terrain_scale),
                  cfg.terrain_map[i][j]) for j in range(cfg.terrain_size)
    ] for i in range(cfg.terrain_size)]
    terrain = Terrain(fuel_arrs)
    environment = Environment(cfg.M_f, cfg.U, cfg.U_dir)

    fire_manager = RothermelFireManager(cfg.fire_init_pos, cfg.fire_size,
                                        cfg.max_fire_duration, cfg.pixel_scale,
                                        fuel_particle, terrain, environment)

    running = True
    while running:
        fire_sprites = fire_manager.sprites
        fire_map = fire_manager.fire_map.copy()
        running = game.update(terrain, fire_sprites, fire_map)
        fire_manager.update()

    pygame.quit()


if __name__ == '__main__':
    main()
