import math
import pathlib
from importlib import resources
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pygame
from PIL import Image

from ..enums import BurnStatus, GameStatus
from ..utils.log import create_logger
from ..utils.units import mph_to_ftpm
from .image import load_image
from .sprites import Agent, Fire, FireLine, Terrain

log = create_logger(__name__)


class Game:
    """
    Class that controls the game. This class will initalize the game and allow for
    terrain, fire, and other sprites to be rendered and interact.
    """

    def __init__(
        self,
        screen_size: Tuple[int, int],
        rescale_size: Optional[Tuple[int, int]] = None,
        headless: bool = False,
        record: bool = False,
        show_wind_magnitude: bool = False,
        show_wind_direction: bool = False,
        mw_speed_min: float = 0.0,
        mw_speed_max: float = mph_to_ftpm(150.0),
        dw_deg_min: float = 0.0,
        dw_deg_max: float = 360.0,
    ) -> None:
        """
        Initalize the class by creating the game display and background.

        Arguments:
            screen_size: The (n,n) size of the game screen/display in pixels.
            rescale_size: The (s, s) size of the display to resize to
            headless: Flag to run in a headless state.
            record: Flag to save a recording of the simulation game screen
        """
        if record and headless:
            raise ValueError(
                "The game cannot be recorded with headless=True. "
                "Got record=True and headless=True"
            )
        self.screen_size = screen_size
        self.rescale_size = rescale_size
        self.headless = headless
        self.record = record
        self.show_wind_magnitude = show_wind_magnitude
        self.show_wind_direction = show_wind_direction
        self.mw_speed_min = mw_speed_min
        self.mw_speed_max = mw_speed_max
        self.dw_deg_min = dw_deg_min
        self.dw_deg_max = dw_deg_max
        # Map to track which pixels are on fire or have burned
        self.fire_map = np.full(screen_size, BurnStatus.UNBURNED)

        self.background: Optional[pygame.surface.Surface] = None
        if not self.headless:
            pygame.init()
            # self.screen is for drawing all backgrounds/sprites at the simulation's scale
            # self.display_screen is for actually displaying the simulation to the user
            # These can have different sizes to do potential rescaling
            if self.rescale_size is None:
                self.display_screen = pygame.display.set_mode(self.screen_size)
            else:
                self.display_screen = pygame.display.set_mode(self.rescale_size)
            self.screen = pygame.Surface(self.screen_size)
            pygame.display.set_caption("SimFire")
            with resources.path("simfire.utils.assets", "fireline_logo.png") as path:
                fireline_logo_path = path
            pygame.display.set_icon(load_image(str(fireline_logo_path)))

            self.background = pygame.Surface(self.screen.get_size())
            self.background = self.background.convert()
            self.background.fill((0, 0, 0))

        self.frames: Optional[List[Image.Image]] = None
        if self.record:
            self.frames = []

    def _toggle_wind_magnitude_display(self):
        """
        Toggle display of wind MAGNITUDE over the main screen.
        """
        self.show_wind_magnitude = not self.show_wind_magnitude
        if self.show_wind_magnitude is False:
            log.info("Wind Magnitude OFF")
        else:
            log.info("Wind Magnitude ON")
        return

    def _toggle_wind_direction_display(self):
        """
        Toggle display of wind DIRECTION over the main screen.
        """
        self.show_wind_direction = not self.show_wind_direction
        if self.show_wind_direction is False:
            log.info("Wind Direction OFF")
        else:
            log.info("Wind Direction ON")
        return

    def _disable_wind_magnitude_display(self):
        """
        Toggle display of wind DIRECTION over the main screen.
        """
        self.show_wind_magnitude = False

    def _disable_wind_direction_display(self):
        """
        Toggle display of wind DIRECTION over the main screen.
        """
        self.show_wind_direction = False

    def _get_wind_direction_color(self, direction: float) -> Tuple[int, int, int]:
        """
        Get the color and intensity representing direction based on wind direction.

        0/360: Black, 90: Red, 180: White, Blue: 270

        Returns tuple of RGBa where a is alpha channel or transparency of the color

        Arguments:
            direction: Float value of the angle 0-360.
            ws_min: Minimum wind speed.
            ws_max: Maximum wind speed.
        """
        north_min = 0.0
        north_max = 360.0
        east = 90.0
        south = 180.0
        west = 270.0

        colorRGB = (255.0, 255.0, 255.0)  # Default white

        # North to East, Red to Green
        if direction >= north_min and direction < east:
            angleRange = east - north_min

            # Red
            redMin = 255.0
            redMax = 128.0
            redRange = redMax - redMin  # 255 - 128 red range from north to east
            red = (((direction - north_min) * redRange) / angleRange) + redMin

            # Green
            greenMin = 0.0
            greenMax = 255.0
            greenRange = greenMax - greenMin  # 0 - 255 red range from north to east
            green = (((direction - north_min) * greenRange) / angleRange) + greenMin

            colorRGB = (red, green, 0.0)

        # East to South, Green to Teal
        if direction >= east and direction < south:
            angleRange = south - east

            # Red
            redMin = 128.0
            redMax = 0.0
            redRange = redMax - redMin  # 128 - 0 red range from east to south
            red = (((direction - east) * redRange) / angleRange) + redMin

            # Blue
            blueMin = 0
            blueMax = 255
            blueRange = blueMax - blueMin  # 0 - 255 blue range from east to south
            blue = (((direction - east) * blueRange) / angleRange) + blueMin

            colorRGB = (red, 255, blue)

        # South to West, Teal to Purple
        if direction >= south and direction < west:
            angleRange = west - south

            # Red
            redMin = 0
            redMax = 128
            redRange = redMax - redMin  # 0 - 128 red range from south to west
            red = (((direction - south) * redRange) / angleRange) + redMin

            # Green
            greenMin = 255
            greenMax = 0
            greenRange = greenMax - greenMin  # 0 - 255 green range from south to west
            green = (((direction - south) * greenRange) / angleRange) + greenMin

            colorRGB = (red, green, 255)

        # West to North, Purple to Red
        if direction <= north_max and direction >= west:
            angleRange = north_max - west

            # Red
            redMin = 128.0
            redMax = 255.0
            redRange = redMax - redMin  # 128 - 255 red range from east to south
            red = (((direction - west) * redRange) / angleRange) + redMin

            # Blue
            blueMin = 0
            blueMax = 255
            blueRange = blueMax - blueMin  # 0 - 255 blue range from east to south
            blue = (((direction - west) * blueRange) / angleRange) + blueMin

            colorRGB = (red, 0, blue)

        floorColorRGB = (
            int(math.floor(colorRGB[0])),
            int(math.floor(colorRGB[1])),
            int(math.floor(colorRGB[2])),
        )
        return floorColorRGB

    def _get_wind_mag_surf(
        self, wind_magnitude_map: Union[Sequence[Sequence[float]], np.ndarray]
    ) -> pygame.surface.Surface:
        """
        Compute the wind magnitude surface for display.

        Arguments:
            wind_magnitude_map: The map/array containing wind magnitudes at each pixel
                                location

        Returns:
            The PyGame Surface for the wind magnitude
        """
        w_max = np.amax(wind_magnitude_map)
        w_min = np.amin(wind_magnitude_map)
        wind_speed_range = w_max - w_min
        # Constant value wind at all locations. Set everything to middle value
        if wind_speed_range == 0:
            color_arr = np.full(self.screen.get_size(), 127, dtype=np.uint8)
            wind_mag_surf = pygame.surfarray.make_surface(color_arr.swapaxes(0, 1))
        else:
            wind_mag_surf = pygame.Surface(self.screen.get_size())
            for y_idx, y in enumerate(wind_magnitude_map):
                for x_idx, x in enumerate(y):
                    w_mag = x
                    gradient = 255 - 0
                    color_mag = int(((w_mag - w_min) * gradient) / (wind_speed_range) + 0)
                    wind_mag_surf.set_at((x_idx, y_idx), pygame.Color(0, color_mag, 0))
        return wind_mag_surf

    def _get_wind_dir_surf(
        self, wind_direction_map: Union[Sequence[Sequence[float]], np.ndarray]
    ) -> pygame.Surface:
        """
        Compute the wind direction surface for display.

        Arguments:
            wind_direction_map: The map/array containing wind directions at each pixel
                                location

        Returns:
            The PyGame Surface for the wind direction
        """
        wind_dir_surf = pygame.Surface(self.screen.get_size())
        for y_idx, y in enumerate(wind_direction_map):
            for x_idx, x in enumerate(y):
                w_dir = x
                color = self._get_wind_direction_color(w_dir)
                pyColor = pygame.Color(color[0], color[1], color[2], a=191)
                wind_dir_surf.set_at((x_idx, y_idx), pyColor)

        return wind_dir_surf

    def quit(self) -> None:
        """
        Close the PyGame window and stop the `Game`
        """
        pygame.display.quit()
        pygame.quit()

    def save(self, path: pathlib.Path, duration: int = 100) -> None:
        """Save a GIF of the simulation to a specified path

        Arguments:
            path: The path to save the GIF to with filename.
            duration: The time to display the current frame of the GIF, in milliseconds.
        """
        if self.frames is not None:
            self.frames[0].save(
                path,
                save_all=True,
                duration=duration,
                loop=0,
                append_images=self.frames[1:],
            )
        else:
            log.error(
                "self.frames is set to None when attempting to save. Make sure "
                "self.frames is not set to None when saving a GIF."
            )
            raise ValueError

    def update(
        self,
        terrain: Terrain,
        fire_sprites: Sequence[Fire],
        fireline_sprites: Sequence[FireLine],
        agent_sprites: Sequence[Agent],
        wind_magnitude_map: Union[Sequence[Sequence[float]], np.ndarray],
        wind_direction_map: Union[Sequence[Sequence[float]], np.ndarray],
    ) -> GameStatus:
        """
        Update the game display using the provided terrain, sprites, and
        environment data. Most of the logic for the game is handled within
        each sprite/manager, so this is mostly just calls to update everything.

        Arguments:
            terrain: The Terrain class that comprises the burnable area.
            fire_sprites: A list of all Fire sprites that are actively burning.
            fireline_sprites: A list of all FireLine sprites that are dug.
            wind_magnitude_map: The map/array containing wind magnitudes at each pixel
                                location
            wind_direction_map: The map/array containing wind directions at each pixel
                                location
        """
        status = GameStatus.RUNNING

        # Convert the sequences to list for list addition later
        if not isinstance(fire_sprites, list):
            fire_sprites = list(fire_sprites)
        if not isinstance(fireline_sprites, list):
            fireline_sprites = list(fireline_sprites)
        if not isinstance(agent_sprites, list):
            agent_sprites = list(agent_sprites)

        if not self.headless:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    status = GameStatus.QUIT

                if event.type == pygame.KEYDOWN:
                    keys = pygame.key.get_pressed()
                    if keys[pygame.K_m] is True:
                        self._disable_wind_direction_display()
                        self._toggle_wind_magnitude_display()

                    if keys[pygame.K_n] is True:
                        self._disable_wind_magnitude_display()
                        self._toggle_wind_direction_display()

        # Create a layered group so that the fire appears on top
        fire_sprites_group = pygame.sprite.LayeredUpdates(
            *(fire_sprites + fireline_sprites)
        )
        agent_sprites_group = pygame.sprite.LayeredUpdates(*agent_sprites)
        all_sprites = pygame.sprite.LayeredUpdates(
            *agent_sprites_group, *fire_sprites_group, terrain
        )

        # Update and draw the sprites
        if not self.headless:
            if self.background is not None:
                for sprite in all_sprites.sprites():
                    if sprite.rect is not None:
                        self.screen.blit(self.background, sprite.rect, sprite.rect)

                fire_sprites_group.update()
                agent_sprites_group.update()
                terrain.update(self.fire_map)
                all_sprites.draw(self.screen)

                if self.rescale_size is None:
                    self.display_screen.blit(self.screen, (0, 0))
                else:
                    self.display_screen.blit(
                        pygame.transform.smoothscale(self.screen, self.rescale_size),
                        (0, 0),
                    )

                if self.record and self.frames is not None:
                    screen_bytes = pygame.image.tostring(self.screen, "RGB")
                    screen_im = Image.frombuffer("RGB", self.screen_size, screen_bytes)
                    self.frames.append(screen_im)

                if self.show_wind_magnitude is True:
                    wind_mag_surf = self._get_wind_mag_surf(wind_magnitude_map)
                    self.screen.blit(wind_mag_surf, (0, 0))

                if self.show_wind_direction is True:
                    wind_dir_surf = self._get_wind_dir_surf(wind_direction_map)
                    self.screen.blit(wind_dir_surf, (0, 0))

                pygame.display.update()

        return status
