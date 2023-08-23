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
from simfire.utils.log import create_logger
from simfire.world.parameters import Environment, FuelParticle

log = create_logger(__name__)


def main():
    """
    Initialize the layers.
    Create the Game, Terrain, and Environment
    Create the Managers
    """

    cfg_path = Path("configs/operational_config.yml")
    log.info(f"Starting simulation with {cfg_path}")
    cfg = Config(cfg_path)

    fuel_particle = FuelParticle()

    if cfg.display.rescale_factor is not None:
        rescale_factor = int(cfg.display.rescale_factor)
    else:
        rescale_factor = None
    log.info(f"Creating game with screen size: {cfg.area.screen_size}")
    log.info(f"Headless: {bool(cfg.simulation.headless)}")
    log.info(f"Recording: {bool(cfg.simulation.record)}")
    game = Game(
        cfg.area.screen_size,
        rescale_factor=rescale_factor,
        headless=cfg.simulation.headless,
        record=cfg.simulation.record,
    )

    log.info("Loading Terrain...")
    terrain = Terrain(
        cfg.terrain.fuel_layer, cfg.terrain.topography_layer, game.screen_size
    )
    log.info("Done loading Terrain")

    log.info("Loading Environment...")
    environment = Environment(
        cfg.environment.moisture, cfg.wind.speed, cfg.wind.direction
    )
    log.info("Done loading environment")

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
    log.info(
        f"Loaded FireLineManager with control line size: {cfg.display.control_line_size}"
    )
    log.info(f"Loaded FireLineManager with pixel scale: {cfg.area.pixel_scale}")

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
        attenuate_line_ros=cfg.mitigation.ros_attenuation,
    )
    fire_manager_str = str(fire_manager).split(".")[-1].split(" ")[0]
    log.info(
        f"Loaded {fire_manager_str} with initial position: "
        f"{cfg.fire.fire_initial_position}"
    )
    log.info(f"Loaded {fire_manager_str} with fire size: {cfg.display.fire_size}")
    log.info(
        f"Loaded {fire_manager_str} with fire duration: {cfg.fire.max_fire_duration}"
    )
    log.info(f"Loaded {fire_manager_str} with update rate: {cfg.simulation.update_rate}")

    game_status = GameStatus.RUNNING
    fire_status = GameStatus.RUNNING
    log.info("Simulation started")
    while game_status == GameStatus.RUNNING and fire_status == GameStatus.RUNNING:
        fire_sprites = fire_manager.sprites
        fireline_sprites = fireline_manager.sprites
        game_status = game.update(
            terrain=terrain,
            fire_sprites=fire_sprites,
            fireline_sprites=fireline_sprites,
            agent_sprites=[],
            wind_magnitude_map=cfg.wind.speed,
            wind_direction_map=cfg.wind.direction,
        )
        fire_map = game.fire_map
        # fire_map = fireline_manager.update(fire_map)
        fire_map, fire_status = fire_manager.update(fire_map)
        game.fire_map = fire_map

    log.info("Simulation fininshed")
    if cfg.simulation.record:
        out_path = Path().cwd() / "simulation.gif"
        log.info(f"Saving {out_path}...")
        game.save(out_path)
        log.info(f"Saved {out_path}")

    if cfg.simulation.headless:
        save_path = Path().cwd() / "fire_spread_graph.png"
        log.info(
            "Game is running in a headless state. Saving fire spread "
            f"graph to {save_path}"
        )

    if cfg.simulation.draw_spread_graph:
        log.info("Drawing fire spread graph...")
        fig = fire_manager.draw_spread_graph(game.screen)
        if cfg.simulation.headless:
            save_path = os.curdir + "fire_spread_graph.png"
            log.info(
                "Game is running in a headless state. Saving fire spread "
                f"graph to {save_path}"
            )
            fig.savefig(save_path)
        else:
            if "DISPLAY" in os.environ:
                log.info(
                    "Game is running in a non-headless state. Displaying fire spread "
                    f'graph on DISPLAY {os.environ["DISPLAY"]}'
                )
                import matplotlib.pyplot as plt

                plt.show()
                while plt.fignum_exists(fig.number):
                    continue


if __name__ == "__main__":
    main()
