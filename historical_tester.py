import os
from datetime import datetime
from pathlib import Path

from src.enums import BurnStatus, GameStatus
from src.game.game import Game
from src.game.managers.fire import RothermelFireManager
from src.game.managers.mitigation import FireLineManager
from src.game.sprites import Terrain
from src.utils.config import Config
from src.utils.units import str_to_minutes
from src.world.parameters import Environment, FuelParticle


def main():
    """
    Initialize the layers.
    Create the Game, Terrain, and Environment
    Create the Managers
    """
    cfg_path = Path("configs/operational_config.yml")
    cfg = Config(cfg_path)

    fuel_particle = FuelParticle()

    game = Game(
        (cfg.area.screen_size, cfg.area.screen_size), record=cfg.simulation.record
    )

    terrain = Terrain(
        cfg.terrain.fuel_layer,
        cfg.terrain.topography_layer,
        game.screen_size,
        hist_layer=cfg.historical_layer,
    )
    # terrain = Terrain(cfg.terrain.fuel_layer, cfg.terrain.topography_layer,
    #                   game.screen_size)

    environment = Environment(
        cfg.environment.moisture, cfg.wind.speed, cfg.wind.direction
    )

    fireline_manager = FireLineManager(
        size=cfg.display.control_line_size,
        pixel_scale=cfg.area.pixel_scale,
        terrain=terrain,
    )

    fire_map = game.fire_map
    fire_map[cfg.fire.fire_initial_position] = BurnStatus.BURNING
    # fire_map = fireline_manager.update(fire_map, points)
    game.fire_map = fire_map

    # fire_manager = RothermelFireManager(
    #     cfg.fire.fire_initial_position,
    #     cfg.display.fire_size,
    #     cfg.fire.max_fire_duration,
    #     cfg.area.pixel_scale,
    #     cfg.simulation.update_rate,
    #     fuel_particle,
    #     terrain,
    #     environment,
    #     max_time=cfg.simulation.runtime,
    # )

    cfg.fire.fire_initial_position = cfg.historical_layer.fire_init_pos

    # run for each historical perimeter duration
    datetimeFormat = "%Y/%m/%d %H:%M:%S.%f"
    time = datetime.strptime(
        cfg.historical_layer.end_time, datetimeFormat
    ) - datetime.strptime(cfg.historical_layer.start_time, datetimeFormat)

    time = time.seconds / 60
    runtime = str(int(time)) + "m"
    cfg.simulation.runtime = str_to_minutes(runtime)

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
            cfg.wind.speed,
            cfg.wind.direction,
        )
        fire_map = game.fire_map
        fire_map = fireline_manager.update(fire_map)
        fire_map, fire_status = fire_manager.update(fire_map)
        game.fire_map = fire_map

    if cfg.simulation.record:
        out_path = os.curdir + "/simulation.gif"
        game.frames[0].save(
            out_path, append_images=game.frames[1:], save_all=True, duration=1000, loop=0
        )


if __name__ == "__main__":
    main()
