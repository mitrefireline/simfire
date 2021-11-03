import pygame
from skimage.draw import line
import random
import src.config as cfg
from src.enums import GameStatus
from src.game.game import Game
from src.game.managers.fire import RothermelFireManager
from src.game.managers.mitigation import FireLineManager
from src.game.sprites import Terrain
from src.world.parameters import Environment, FuelArray, FuelParticle, Tile
from src.rl_env.fireline_env import FireLineEnv, RLEnv


def main():

    game = Game(cfg.screen_size)

    fuel_particle = FuelParticle()

    fuel_arrs = [[
        FuelArray(Tile(j, i, cfg.terrain_scale, cfg.terrain_scale), cfg.terrain_map[i][j])
        for j in range(cfg.terrain_size)
    ] for i in range(cfg.terrain_size)]
    terrain = Terrain(fuel_arrs, cfg.elevation_fn, cfg.terrain_size, cfg.screen_size)
    environment = Environment(cfg.M_f, cfg.U, cfg.U_dir)

    points = line(100, 15, 100, 200)
    y = points[0].tolist()
    x = points[1].tolist()
    points = list(zip(x, y))

    fireline_manager = FireLineManager(size=cfg.control_line_size,
                                       pixel_scale=cfg.pixel_scale,
                                       terrain=terrain)

    fire_map = game.fire_map
    fire_map = fireline_manager.update(fire_map, points)
    game.fire_map = fire_map

    fire_manager = RothermelFireManager(cfg.fire_init_pos, cfg.fire_size,
                                        cfg.max_fire_duration, cfg.pixel_scale,
                                        fuel_particle, terrain, environment)

    game_status = GameStatus.RUNNING
    fire_status = GameStatus.RUNNING
    while game_status == GameStatus.RUNNING and fire_status == GameStatus.RUNNING:
        fire_sprites = fire_manager.sprites
        fireline_sprites = fireline_manager.sprites
        game_status = game.update(terrain, fire_sprites, fireline_sprites)
        fire_map = game.fire_map
        fire_map = fireline_manager.update(fire_map)
        fire_map, fire_status = fire_manager.update(fire_map)
        game.fire_map = fire_map


def some_action_func(state):
    '''
    A dummy function to show how the rl side ingests the state
        and returns a dict() of the fire mitigation stategy
    '''
    fire_mitigation = random.choices([0, 1], weights=[0.8, 0.2])
    return fire_mitigation[0]


def rl_main():

    env_config = FireLineEnv(cfg)
    rl_environment = RLEnv(env_config)
    state = rl_environment.reset()
    done = False
    final_reward = 0
    while not done:
        action = some_action_func(state)
        state, reward, done, _ = rl_environment.step(action)
        final_reward += reward

    pygame.quit()


if __name__ == '__main__':
    # main()
    rl_main()
