import pygame
from random import uniform

from src.game.viewer import ProjectionViewer
from src.world.parameters import Environment, FuelArray, Tile
from src.world.terrain import Terrain


def main() -> None:
    # Create the environment
    M_f = 0.5
    U = 88 # ft/min or 1mph
    environment = Environment(M_f, U)
    # Make each tile and fuel array the same
    x = 1
    y = 1
    z = 0
    w_0 = 1
    delta = 1
    M_x = 0.5
    terrain_shape = (30, 30)
    arrays = [[FuelArray(Tile(x, y, uniform(0, 3)), w_0, delta, M_x) for j in range(terrain_shape[1])] for i in range(terrain_shape[0])]
    terrain = Terrain(environment, arrays)
    pv = ProjectionViewer(400, 300)
    pv.addWireframe('terrain', terrain)
    pv.run()


if __name__ == '__main__':
    main()