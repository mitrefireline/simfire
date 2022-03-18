
import pygame
# import fluid
import numpy as np
import random
import time
from pathlib import Path
from src.utils.config import Config
from src.world.wind_mechanics.wind_controller import WindController2
pygame.init()

# Seperate file to precompute wind layer, currently using wind with a single time slice due
# to processing limitations


def renderD(surface, screen_size, scale, density) -> None:
    for i in range(0, screen_size):
        for j in range(0, screen_size):
            x = i * scale
            y = j * scale
            d = density[i][j]
            if d > 255:
                d = 255
            rect = pygame.Rect(x, y, scale, scale)
            pygame.draw.rect(surface, (d, d, d), rect)
    return


def renderV(surface, screen_size, scale, velocity_x, velocity_y) -> None:
    for i in range(0, screen_size):
        for j in range(0, screen_size):
            x = i * scale
            y = j * scale
            vx = velocity_x[i][j]
            vy = velocity_y[i][j]
            if (not (abs(vx) < 0.1 and abs(vy)) <= 0.1):
                meanval = int(np.mean([vx, vy]))
                if meanval < 0:
                    meanval = 0
                if meanval > 255:
                    meanval = 255
                pygame.draw.line(surface, [meanval, meanval, meanval, meanval],
                                 [x, y], [x + vx, y + vy])
    return


def generate_cfd_wind_layer(display: bool = False):
    time_end = time.time() + 10
    cfg_path = Path('./config.yml')
    cfg = Config(cfg_path, cfd_precompute=True)
    wind_map = cfg.get_cfd_wind_map()

    wm_scale = wind_map.get_wind_scale()
    wm_size = wind_map.get_screen_size()

    if display is True:
        screen = pygame.display.set_mode([wm_size, wm_size])
        screen.fill('white')
        pygame.display.flip()

    while time.time() < time_end:
        wind_map.iterate_wind_step()
        # wm_density = wind_map.get_wind_density_field()
        wm_velocity_x = wind_map.get_wind_velocity_field_x()
        wm_velocity_y = wind_map.get_wind_velocity_field_y()

        wind_map.fvect.step()
        if display is True:
            # renderD(screen, wm_size, wm_scale, wm_density)
            renderV(screen, wm_size, wm_scale, wm_velocity_x, wm_velocity_y)
            pygame.display.flip()

    np.save('generated_wind_velocity_map_x', wm_velocity_x)
    np.save('generated_wind_velocity_map_y', wm_velocity_y)


if __name__ == '__main__':
    #x = np.load('generated_wind_velocity_map_x.npy')
    #y = np.load('generated_wind_velocity_map_y.npy')
    generate_cfd_wind_layer(False)
