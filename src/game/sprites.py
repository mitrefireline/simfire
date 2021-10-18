from typing import Sequence, Tuple

import matplotlib.contour as mcontour
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from PIL import Image, ImageDraw
import pygame

from .image import load_image
from .. import config as cfg
from ..enums import BurnStatus, DRY_TERRAIN_BROWN_IMG, FIRE_TEXTURE_PATH, SpriteLayer,\
    TERRAIN_TEXTURE_PATH, FIRELINE_TEXTURE_PATH, BURNED_RGB_COLOR
from ..world.elevation_functions import ElevationFn
from ..world.parameters import FuelArray


class Terrain(pygame.sprite.Sprite):
    '''
    Use the FuelArray tiles to make terrain. This sprite is just the
    entire background that the fire appears on. This sprite stitches the
    tiles together initially and then updates their color based on burn
    status.
    '''
    def __init__(self, tiles: Sequence[Sequence[FuelArray]],
                 elevation_fn: ElevationFn) -> None:
        '''
        Initialize the class by loading the tile textures and stitching
        together the whole terrain image.

        Arguments:
            tiles: The 2D nested sequences of FuelArrays that comprise the
                   terrain.
            elevation_fn: A callable function that converts (x, y) coordintates to
                          elevations

        Returns:
            None
        '''
        super().__init__()

        self.tiles = np.array(tiles)
        self.terrain_size = cfg.terrain_size
        self.elevation_fn = elevation_fn

        self.screen_size = (cfg.screen_size, cfg.screen_size)
        self.texture = self._load_texture()
        self.image, self.fuel_arrs, self.elevations = self._make_terrain_image()
        # The rectangle for this sprite is the entire game
        self.rect = pygame.Rect(0, 0, *self.screen_size)

        # This sprite should always have layer 1 since it will always
        # be behind every other sprite
        self.layer = SpriteLayer.TERRAIN

    def update(self, fire_map: np.ndarray) -> None:
        '''
        Change any burned squares to brown using fire_map, which
        contains pixel-wise values for tile burn status.

        Arguments:
            fire_map: A map containing enumerated values for unburned,
                      burning, and burned tile status

        Returns: None
        '''
        fire_map = fire_map.copy()
        burned_idxs = np.where(fire_map == BurnStatus.BURNED)
        # This method will update self.image in-place with arr
        arr = pygame.surfarray.pixels3d(self.image)
        arr[burned_idxs[::-1]] = BURNED_RGB_COLOR

    def _load_texture(self) -> np.ndarray:
        '''
        Load the terrain tile texture, resize it to the correct
        shape, and convert to numpy

        Arguments:
            None

        Returns:
            None
        '''
        out_size = (self.terrain_size, self.terrain_size)
        texture = Image.open(TERRAIN_TEXTURE_PATH)
        texture = texture.resize(out_size)
        texture = np.array(texture)

        return texture

    def _make_terrain_image(self) -> Tuple[pygame.Surface, np.ndarray, np.ndarray]:
        '''
        Stitch together all of the FuelArray tiles in self.tiles to create
        the terrain image. This starts as a numpy array, but is then converted
        to a pygame.Surface for compatibility with pygame.

        Additionally, stitch together the FuelArrays into a tiled
        numpy array that aligns with out_surf for use with a FireManager.

        Additionally, create the elevation array that contains the elevation
        for each pixel in self.fuel_arrs

        Arguments:
            None

        Returns:
            out_surf: The pygame.Surface of the stitched together terrain
                      tiles and contour lines
            fuel_arrs: A (screen_size x screen_size) array containing the
                       FuelArray data at the pixel level. This allows for finer
                       resolution for the FireManager to work at the pixel
                       level
            elevations: A (screen_size x screen_size) array containing the elevation at
                        the pixel level
        '''
        image = np.zeros(self.screen_size + (3, ))
        fuel_arrs = np.zeros(self.screen_size, dtype=np.dtype(FuelArray))

        # Loop over the high-level tiles (these are not at the pixel level)
        for i in range(self.tiles.shape[0]):
            for j in range(self.tiles.shape[1]):
                # Need tese pixel level coordinates to span the correct range
                x = j * self.terrain_size
                y = i * self.terrain_size
                w = self.terrain_size
                h = self.terrain_size

                updated_texture = self._update_texture_dryness(self.tiles[j][i])
                image[y:y + h, x:x + w] = updated_texture
                fuel_arrs[y:y + h, x:x + w] = self.tiles[i][j]

        image, elevations = self._make_contour_image(image, fuel_arrs)
        out_surf = pygame.surfarray.make_surface(image)

        return out_surf, fuel_arrs, elevations

    def _make_contour_image(self, image: np.ndarray, fuel_arrs: np.ndarray) -> \
                            Tuple[np.ndarray, np.ndarray]:
        '''
        Use the image, FuelArray, and elevation_fn to create the elevations array and
        compute the contours. The contours are computed with plt.contours, and the contour
        lines are drawn by converting image to a PIL.Image.Image and using the ImageDraw
        module.

        Arguments:
            image: A numpy array representing the np.float RGB terrain image for display
            fuel_arrs: A (screen_size x screen_size) array containing the
                       FuelArray data at the pixel level

        Returns:
            out_image: The input image with the contour lines drawn on it
            z: A (screen_size x screen_size) array containing the elevation at
               the pixel level
        '''
        # Create a meshgrid to compute the elevations at all pixel points
        x = np.arange(fuel_arrs.shape[1])
        y = np.arange(fuel_arrs.shape[0])
        X, Y = np.meshgrid(x, y)
        fn = np.vectorize(self.elevation_fn)
        z = fn(X, Y)

        # Convert the image to a PIL.Image.Image and draw on it
        img = Image.fromarray(image.astype(np.uint8))
        draw = ImageDraw.Draw(img)
        # Use a static number of levels
        levels = np.linspace(np.min(z), np.max(z), 20)
        cont = mcontour.QuadContourSet(plt.gca(), X, Y, z, levels=levels)

        # Use a diverging colormap so that higher elevations are green and lower
        # elevations are purple
        cmap = cm.get_cmap('PRGn')

        # Loop over all contours and their levels to draw them
        for level, segs in zip(cont.levels, cont.allsegs):
            if segs == []:
                continue
            seg = segs[0]
            r = (level - cont.zmin) / (cont.zmax - cont.zmin)
            # Remove the alpha value
            icmap = cmap(r)[:3]
            # Normalize to [0, 255] and convert to uint8 for Image display
            icmap = (255 * np.array(icmap)).astype(np.uint8)
            # The segs are returned in a numpy array of shape (num_points, 2)
            # Map them to tuples for compatibility with ImageDraw
            coords = tuple(map(tuple, seg))
            draw.line(coords, fill=tuple(icmap.tolist()), width=1)
            text_loc = seg[seg.shape[0] // 2]
            draw.text(text_loc.tolist(), f'{int(level)}', stroke_width=1, direction='rtl')

        # Convert to the desired output format
        out_image = np.array(img).astype(np.float32)

        return out_image, z

    def _update_texture_dryness(self, fuel_arr: FuelArray) -> np.ndarray:
        '''
        Determine the percent change to make the terrain look drier (i.e.
        more red/yellow/brown) by using the FuelArray values. Then, update
        the texture color using PIL and image blending with a preset
        yellow-brown color/image.

        Arguments:
            fuel_arr: The FuelArray with parameters that specify how
                      "dry" the texture should look

        Returns:
            new_texture: The texture with RGB calues modified to look drier based
                         on the parameters of fuel_arr
        '''
        # Add the numbers after normalization
        # M_x is inverted because a lower value is more flammable
        color_change_pct = fuel_arr.fuel.w_0 / 0.2296 + \
                           fuel_arr.fuel.delta / 7 + \
                           (0.2 - fuel_arr.fuel.M_x) / 0.2
        # Divide by 3 since there are 3 values
        color_change_pct /= 3

        arr = self.texture.copy()
        arr_img = Image.fromarray(arr)
        texture_img = Image.blend(arr_img, DRY_TERRAIN_BROWN_IMG, color_change_pct / 2)
        new_texture = np.array(texture_img)

        return new_texture


class Fire(pygame.sprite.Sprite):
    '''
    This sprite represents a fire burning on one pixel of the terrain. Its
    image is generally kept very small to make rendering easier. All fire
    spreading is handled by the FireManager it is attached to.
    '''
    def __init__(self, pos: Tuple[int, int], size: int) -> None:
        '''
        Initialize the class by recording the position and size of the sprite
        and loading/resizing its texture.

        Arguments:
            pos: The (x, y) pixel position of the sprite
            size: The pixel size of the sprite
        '''
        super().__init__()

        self.pos = pos
        self.size = size

        self.image = load_image(FIRE_TEXTURE_PATH)
        self.image = pygame.transform.scale(self.image, (self.size, self.size))

        self.rect = self.image.get_rect()
        self.rect = self.rect.move(self.pos[0], self.pos[1])

        # Layer 3 so that it appears on top of the terrain and line (if applicable)
        self.layer: int = SpriteLayer.FIRE

        # Record how many frames this sprite has been alive
        self.duration: int = 0

    def update(self) -> None:
        '''
        Check which squares the fire is on and adjacent to and update its
        spread.

        Arguments:
            None

        Returns:
            None
        '''
        # Increment the duration
        self.duration += 1


class FireLine(pygame.sprite.Sprite):
    '''
    This sprite represents a fire burning on one pixel of the terrain. Its
    image is generally kept very small to make rendering easier. All fire
    spreading is handled by the FireManager it is attached to.
    '''
    def __init__(self, pos: Tuple[int, int], size: int) -> None:
        '''
        Initialize the class by recording the position and size of the sprite
        and loading/resizing its texture
        '''
        super().__init__()

        self.pos = pos
        self.size = size

        self.image = load_image(FIRELINE_TEXTURE_PATH)
        self.image = pygame.transform.scale(self.image, (self.size, self.size))

        self.rect = self.image.get_rect()
        self.rect = self.rect.move(self.pos[0], self.pos[1])

        # Layer 2 so that it appears on top of the terrain
        self.layer: int = SpriteLayer.LINE

        # Record how many frames this sprite has been alive

    def update(self) -> None:
        '''
        Check which squares the fire is on and adjacent to and update its
        spread.

        Arguments:
            None

        Returns:
            None
        '''
        pass
