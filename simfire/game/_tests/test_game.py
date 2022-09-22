import os
import unittest
from multiprocessing import get_context

from PIL import Image

from ...enums import GameStatus
from ...utils.config import Config
from ...world.parameters import Environment, FuelParticle
from ..game import Game
from ..managers.fire import RothermelFireManager
from ..managers.mitigation import FireLineManager
from ..sprites import Terrain

# unittest.mock.patch.dict isn't working anymore
# Can't run rests without setting display at the top of the file
os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame  # noqa: E402


class TestGame(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config("./simfire/utils/_tests/test_configs/test_config.yml")
        self.screen_size = (self.config.area.screen_size, self.config.area.screen_size)
        self.game = Game(self.screen_size)

    def test__toggle_wind_magnitude_display(self) -> None:
        """
        Test that when function is called, `game.show_wind_magnitude` is inverted
        """
        self.game.show_wind_magnitude = False
        self.game._toggle_wind_magnitude_display()
        self.assertTrue(
            self.game.show_wind_magnitude,
            msg="Game().show_wind_magnitude was not toggled from False to " "True",
        )
        self.game._toggle_wind_magnitude_display()
        self.assertFalse(
            self.game.show_wind_magnitude,
            msg="Game().show_wind_magnitude was not toggled from True to " "False",
        )

    def test__toggle_wind_direction_display(self) -> None:
        """
        Test that when function is called, `game.show_wind_direction` is inverted
        """
        self.game.show_wind_direction = False
        self.game._toggle_wind_direction_display()
        self.assertTrue(
            self.game.show_wind_direction,
            msg="Game().show_wind_direction was not toggled from False to " "True",
        )
        self.game._toggle_wind_direction_display()
        self.assertFalse(
            self.game.show_wind_direction,
            msg="Game().show_wind_direction was not toggled from True to " "False",
        )

    def test__disable_wind_magnitude_display(self) -> None:
        """
        Test that when function is called, `game.show_wind_magnitude` is disabled
        """
        self.game.show_wind_magnitude = True
        self.game._disable_wind_magnitude_display()
        self.assertFalse(
            self.game.show_wind_magnitude,
            msg="Game().show_wind_magnitude was not disabled and changed "
            "from True to False",
        )

    def test__disable_wind_direction_display(self) -> None:
        """
        Test that when function is called, `game.show_wind_direction` is disabled
        """
        self.game.show_wind_direction = True
        self.game._disable_wind_direction_display()
        self.assertFalse(
            self.game.show_wind_direction,
            msg="Game().show_wind_direction was not disabled and changed "
            "from True to False",
        )

    def test__get_wind_direction_color(self) -> None:
        """
        Test getting the color of the wind direction
        """
        # North
        direction = 0.0
        rgb = (255, 0, 0)
        returned_rgb = self.game._get_wind_direction_color(direction)
        self.assertEqual(
            rgb,
            returned_rgb,
            msg=f"Direction angle of {direction} should return color of "
            f"{rgb} when {returned_rgb} was returned",
        )
        # East
        direction = 90.0
        rgb = (128, 255, 0)
        returned_rgb = self.game._get_wind_direction_color(direction)
        self.assertEqual(
            rgb,
            returned_rgb,
            msg=f"Direction angle of {direction} should return color of "
            f"{rgb} when {returned_rgb} was returned",
        )
        # South
        direction = 180.0
        rgb = (0, 255, 255)
        returned_rgb = self.game._get_wind_direction_color(direction)
        self.assertEqual(
            rgb,
            returned_rgb,
            msg=f"Direction angle of {direction} should return color of "
            f"{rgb} when {returned_rgb} was returned",
        )
        # West
        direction = 270.0
        rgb = (128, 0, 0)
        returned_rgb = self.game._get_wind_direction_color(direction)
        self.assertEqual(
            rgb,
            returned_rgb,
            msg=f"Direction angle of {direction} should return color of "
            f"{rgb} when {returned_rgb} was returned",
        )

    def test__get_wind_mag_surf(self) -> None:
        """
        Test getting the wind magnitude PyGame surface
        """
        surface = self.game._get_wind_mag_surf(self.config.wind.speed)
        surface_size = surface.get_size()
        self.assertIsInstance(
            surface,
            pygame.Surface,
            msg="The object returned from Game()._get_wind_mag_surf "
            f"is a {type(surface)} when it should be a pygame.Surface",
        )
        self.assertEqual(
            surface_size,
            self.screen_size,
            msg="The size of the surface returned in "
            f"Game()._get_wind_mag_surf is {surface_size} when it should be "
            f"{self.screen_size}",
        )

    def test__get_wind_dir_surf(self) -> None:
        """
        Test getting the wind direction PyGame surface
        """
        surface = self.game._get_wind_dir_surf(self.config.wind.direction)
        surface_size = surface.get_size()
        self.assertIsInstance(
            surface,
            pygame.Surface,
            msg="The object returned from Game()._get_wind_dir_surf "
            f"is a {type(surface)} when it should be a pygame.Surface",
        )
        self.assertEqual(
            surface_size,
            self.screen_size,
            msg="The size of the surface returned in "
            f"Game()._get_wind_dir_surf is {surface_size} when it should be "
            f"{self.screen_size}",
        )

    def test_non_headless_update(self) -> None:
        """
        Test that the call to update() runs through properly. There's not much to check
        since the update method only calls sprite and manager update methods. In theory,
        if all the other unit tests pass, then this one should pass.
        """
        headless = False
        fuel_particle = FuelParticle()
        terrain = Terrain(
            self.config.terrain.fuel_layer,
            self.config.terrain.topography_layer,
            self.screen_size,
            headless=headless,
        )
        # Use simple/constant wind speed
        environment = Environment(
            self.config.environment.moisture,
            self.config.wind.speed,
            self.config.wind.direction,
        )
        fireline_manager = FireLineManager(
            size=self.config.display.control_line_size,
            pixel_scale=self.config.area.pixel_scale,
            terrain=terrain,
            headless=headless,
        )
        fire_init_pos = (self.screen_size[0] // 2, self.screen_size[1] // 2)
        fire_manager = RothermelFireManager(
            fire_init_pos,
            self.config.display.fire_size,
            self.config.fire.max_fire_duration,
            self.config.area.pixel_scale,
            self.config.simulation.update_rate,
            fuel_particle,
            terrain,
            environment,
            max_time=self.config.simulation.runtime,
            headless=headless,
        )
        agent_sprites = []
        status = self.game.update(
            terrain,
            fire_manager.sprites,
            fireline_manager.sprites,
            agent_sprites,
            self.config.wind.speed,
            self.config.wind.direction,
        )
        self.assertEqual(
            status,
            GameStatus.RUNNING,
            msg=(
                f"The returned status of the game is {status}, but it "
                f"should be {GameStatus.RUNNING}"
            ),
        )


class TestHeadlessGame(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config("./simfire/utils/_tests/test_configs/test_config.yml")
        self.screen_size = (self.config.area.screen_size, self.config.area.screen_size)
        self.headless = True
        self.game = Game(self.screen_size, headless=self.headless)
        return super().setUp()

    def test_single_process(self) -> None:
        """
        Test that the game can run in a headless state with no PyGame assets loaded.
        This will also allow for the game to be pickle-able and used with multiprocessing.
        """
        fuel_particle = FuelParticle()
        terrain = Terrain(
            self.config.terrain.fuel_layer,
            self.config.terrain.topography_layer,
            self.screen_size,
            headless=self.headless,
        )
        # Use simple/constant wind speed
        environment = Environment(
            self.config.environment.moisture,
            self.config.wind.speed,
            self.config.wind.direction,
        )
        fireline_manager = FireLineManager(
            size=self.config.display.control_line_size,
            pixel_scale=self.config.area.pixel_scale,
            terrain=terrain,
            headless=self.headless,
        )
        fire_init_pos = (self.screen_size[0] // 2, self.screen_size[1] // 2)
        fire_manager = RothermelFireManager(
            fire_init_pos,
            self.config.display.fire_size,
            self.config.fire.max_fire_duration,
            self.config.area.pixel_scale,
            self.config.simulation.update_rate,
            fuel_particle,
            terrain,
            environment,
            max_time=self.config.simulation.runtime,
            headless=self.headless,
        )
        agent_sprites = []
        status = self.game.update(
            terrain,
            fire_manager.sprites,
            fireline_manager.sprites,
            agent_sprites,
            self.config.wind.speed,
            self.config.wind.direction,
        )
        self.assertEqual(
            status,
            GameStatus.RUNNING,
            msg=(
                f"The returned status of the game is {status}, but it "
                f"should be {GameStatus.RUNNING}"
            ),
        )


class TestMultiprocessGame(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config("./simfire/utils/_tests/test_configs/test_config.yml")
        self.screen_size = (self.config.area.screen_size, self.config.area.screen_size)
        self.headless = True
        self.game = Game(self.screen_size, headless=self.headless)
        return super().setUp()

    def test_multiprocess(self) -> None:
        """
        Test that the game will work with multiprocessing
        """
        fuel_particle = FuelParticle()
        terrain = Terrain(
            self.config.terrain.fuel_layer,
            self.config.terrain.topography_layer,
            self.screen_size,
            headless=self.headless,
        )
        # Use simple/constant wind speed
        environment = Environment(
            self.config.environment.moisture,
            self.config.wind.speed,
            self.config.wind.direction,
        )
        fireline_manager = FireLineManager(
            size=self.config.display.control_line_size,
            pixel_scale=self.config.area.pixel_scale,
            terrain=terrain,
            headless=self.headless,
        )
        fire_init_pos = (self.screen_size[0] // 2, self.screen_size[1] // 2)
        fire_manager = RothermelFireManager(
            fire_init_pos,
            self.config.display.fire_size,
            self.config.fire.max_fire_duration,
            self.config.area.pixel_scale,
            self.config.simulation.update_rate,
            fuel_particle,
            terrain,
            environment,
            max_time=self.config.simulation.runtime,
            headless=self.headless,
        )
        pool_size = 1
        agent_sprites = []
        inputs = (
            terrain,
            fire_manager.sprites,
            fireline_manager.sprites,
            agent_sprites,
            self.config.wind.speed,
            self.config.wind.direction,
        )
        inputs = [inputs] * pool_size
        with get_context("spawn").Pool(pool_size) as p:
            status = p.starmap(self.game.update, inputs)

        valid_status = [GameStatus.RUNNING] * pool_size
        self.assertCountEqual(
            status,
            valid_status,
            msg=(
                f"The returned status of the games is {status}, but "
                f"should be {valid_status}"
            ),
        )


class TestRecordingGame(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config("./simfire/utils/_tests/test_configs/test_config.yml")
        self.screen_size = (self.config.area.screen_size, self.config.area.screen_size)
        self.headless = False
        self.game = Game(self.screen_size, headless=self.headless, record=True)
        return super().setUp()

    def test_record(self) -> None:
        """
        Test that the game will record all frames
        """
        fuel_particle = FuelParticle()
        terrain = Terrain(
            self.config.terrain.fuel_layer,
            self.config.terrain.topography_layer,
            self.screen_size,
            headless=self.headless,
        )
        # Use simple/constant wind speed
        environment = Environment(
            self.config.environment.moisture,
            self.config.wind.speed,
            self.config.wind.direction,
        )
        fireline_manager = FireLineManager(
            size=self.config.display.control_line_size,
            pixel_scale=self.config.area.pixel_scale,
            terrain=terrain,
            headless=self.headless,
        )
        fire_init_pos = (self.screen_size[0] // 2, self.screen_size[1] // 2)
        fire_manager = RothermelFireManager(
            fire_init_pos,
            self.config.display.fire_size,
            self.config.fire.max_fire_duration,
            self.config.area.pixel_scale,
            self.config.simulation.update_rate,
            fuel_particle,
            terrain,
            environment,
            max_time=self.config.simulation.runtime,
            headless=self.headless,
        )
        agent_sprites = []
        status = self.game.update(
            terrain,
            fire_manager.sprites,
            fireline_manager.sprites,
            agent_sprites,
            self.config.wind.speed,
            self.config.wind.direction,
        )

        self.assertEqual(
            status,
            GameStatus.RUNNING,
            msg=(
                f"The returned status of the game is {status}, but it "
                f"should be {GameStatus.RUNNING}"
            ),
        )

        frames = self.game.frames
        self.assertEqual(
            len(frames), 1, msg=f"There should be 1 frame recorded, but got {len(frames)}"
        )
        self.assertIsInstance(
            frames[0],
            Image.Image,
            msg="The returned frame should be a numpy array, "
            f"but got {type(frames[0])}",
        )
        frame_size = frames[0].size
        valid_shape = self.screen_size
        self.assertEqual(
            frame_size,
            valid_shape,
            msg=f"The returned frame should have shape {valid_shape}, "
            f"but has shape {frame_size}",
        )
