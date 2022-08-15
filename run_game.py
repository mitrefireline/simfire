import os
from pathlib import Path

import numpy as np
from skimage.draw import line

from simfire.enums import BurnStatus, GameStatus
from simfire.game.game import Game
from simfire.game.managers.fire import RothermelFireManager
from simfire.game.managers.mitigation import FireLineManager
from simfire.game.sprites import Terrain
from simfire.utils.config import Config
from simfire.world.parameters import Environment, FuelParticle


def main():
    """
    Initialize the layers.
    Create the Game, Terrain, and Environment
    Create the Managers
    """
    cfg_path = Path("configs/operational_config.yml")
    cfg = Config(cfg_path)

    fuel_particle = FuelParticle()

    if cfg.display.rescale_size is not None:
        rescale_size = (cfg.display.rescale_size, cfg.display.rescale_size)
    else:
        rescale_size = None
    game = Game(
        (cfg.area.screen_size, cfg.area.screen_size),
        rescale_size=rescale_size,
        record=cfg.simulation.record,
    )

    terrain = Terrain(
        cfg.terrain.fuel_layer, cfg.terrain.topography_layer, game.screen_size
    )

    environment = Environment(
        cfg.environment.moisture, cfg.wind.speed, cfg.wind.direction
    )

    # Need to create two lines to "double up" since the fire can spread
    # to 8-connected squares
    r0, c0 = (0, game.screen_size[1] // 4)
    r1, c1 = (game.screen_size[1] // 4, 0)
    points1 = line(r0, c0, r1, c1)
    r0, c0 = (game.screen_size[1] // 4 - 1, 0)
    r1, c1 = (0, game.screen_size[1] // 4 - 1)
    points2 = line(r0, c0, r1, c1)
    points = tuple(np.concatenate((p1, p2)) for p1, p2 in zip(points1, points2))

    y = points[0].tolist()
    x = points[1].tolist()
    points = list(zip(x, y))

    fireline_manager = FireLineManager(
        size=cfg.display.control_line_size,
        pixel_scale=cfg.area.pixel_scale,
        terrain=terrain,
    )

    fire_map = game.fire_map
    fire_map[cfg.fire.fire_initial_position] = BurnStatus.BURNING
    fire_map = fireline_manager.update(fire_map, points)
    game.fire_map = fire_map

    fire_manager = RothermelFireManager(
        cfg.fire.fire_initial_position,
        cfg.display.fire_size,
        cfg.fire.max_fire_duration,
        cfg.area.pixel_scale,
        cfg.simulation.update_rate,
        fuel_particle,
        terrain,
        environment,
        max_time=cfg.simulation.runtime,
    )

    game_status = GameStatus.RUNNING
    fire_status = GameStatus.RUNNING
    while game_status == GameStatus.RUNNING and fire_status == GameStatus.RUNNING:
        fire_sprites = fire_manager.sprites
        fireline_sprites = fireline_manager.sprites
        game_status = game.update(
            terrain,
            fire_sprites,
            fireline_sprites,
            [],
            cfg.wind.speed,
            cfg.wind.direction,
        )
        fire_map = game.fire_map
        fire_map = fireline_manager.update(fire_map)
        fire_map, fire_status = fire_manager.update(fire_map)
        game.fire_map = fire_map

    if cfg.simulation.record:
        out_path = Path().cwd() / "simulation.gif"
        game.save(out_path)

    fig = fire_manager.draw_spread_graph(game.screen)
    if cfg.simulation.headless:
        save_path = Path().cwd() / "fire_spread_graph.png"
        print(
            "Game is running in a headless state. Saving fire spread "
            f"graph to {save_path}"
        )

    if cfg.simulation.draw_spread_graph:
        fig = fire_manager.draw_spread_graph(game.screen)
        if cfg.simulation.headless:
            save_path = os.curdir + "fire_spread_graph.png"
            print(
                "Game is running in a headless state. Saving fire spread "
                f"graph to {save_path}"
            )
            fig.savefig(save_path)
        else:
            if "DISPLAY" in os.environ:
                print(
                    "Game is running in a non-headless state. Displaying fire spread "
                    f'graph on DISPLAY {os.environ["DISPLAY"]}'
                )
                import matplotlib.pyplot as plt

                plt.show()
                while plt.fignum_exists(fig.number):
                    continue


if __name__ == "__main__":
    main()
