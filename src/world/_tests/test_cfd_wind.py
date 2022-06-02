import unittest

from ...game._tests import DummyFuelLayer, DummyTopographyLayer
from ...game.sprites import Terrain
from ..wind_mechanics.cfd_wind import Fluid


class TestFluid(unittest.TestCase):
    def setUp(self) -> None:
        self.n = 10
        self.shape = (self.n, self.n)
        self.iterations = 1
        self.scale = 400
        self.dt = 1
        self.diffusion = 0.0
        self.viscosity = 0.0000001
        self.topo_layer = DummyTopographyLayer(self.shape)
        self.fuel_layer = DummyFuelLayer(self.shape)
        self.terrain = Terrain(
            self.fuel_layer, self.topo_layer, self.shape, headless=True
        )
        self.fluid = Fluid(
            self.n,
            self.iterations,
            self.scale,
            self.dt,
            self.diffusion,
            self.viscosity,
            self.terrain.topo_layer.data,
        )
        return super().setUp()

    def test_addDensity(self) -> None:
        """Test that adding a density amount is reflected in the class"""
        x = 1
        y = 1
        amount = 0.2
        self.fluid.addDensity(x, y, amount)

        self.assertEqual(
            self.fluid.density[x][y],
            amount,
            msg="The fluid density at the changed coordinates is "
            f"{self.fluid.density[x][y]}, but should be {amount}",
        )

    def test_addVelocity(self) -> None:
        """Test that adding velocity amounts is reflected in the class"""
        x = 1
        y = 1
        amount_x = 0.2
        amount_y = 0.3
        self.fluid.addVelocity(x, y, amount_x, amount_y)

        self.assertEqual(
            self.fluid.Vx[x][y],
            amount_x,
            msg="The fluid x-veloicty at the changed coordinates is "
            f"{self.fluid.Vx[x][y]}, but should be {amount_x}",
        )
        self.assertEqual(
            self.fluid.Vy[x][y],
            amount_y,
            msg="The fluid y-veloicty at the changed coordinates is "
            f"{self.fluid.Vy[x][y]}, but should be {amount_y}",
        )

    def test_step(self) -> None:
        """Test that the step function runs with no errors"""
        self.assertIsNone(self.fluid.step())
