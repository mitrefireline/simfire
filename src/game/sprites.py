import tempfile
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pygame

from ..enums import (BurnStatus, SpriteLayer, BURNED_RGB_COLOR)
from ..utils.layers import DataLayer


class Terrain(pygame.sprite.Sprite):
    '''
    Use a TopographyLayer and a FuelLayer to make terrain. This sprite is just the
    entire background that the fire appears on. This sprite changes the color of each
    terrain pixel based on its "dryness/flammability" if it is unburned, or based on
    whether the pixel is a fireline or burned.
    '''
    def __init__(self,
                 fuel_layer: DataLayer,
                 topo_layer: DataLayer,
                 screen_size: Tuple[int, int],
                 headless: bool = False) -> None:
        '''
        Initialize the class by loading the tile textures and stitching
        together the whole terrain image.

        Arguments:
            fuel_layer: A FuelLayer containing a `data` parameter that is a numpy array
                        containing a Fuel objects
            topo_layer: A TopographyLayer that desctibes the topography (elevation) of the
                        terrain as a numpy array in it `data` parameter
            screen_size: The game's screen size in pixels
            headless: Flag to run in a headless state. This will allow PyGame objects to
                      not be initialized.

        '''
        super().__init__()

        self.fuel_layer = fuel_layer
        self.topo_layer = topo_layer

        self.screen_size = screen_size
        # self.texture = self._load_texture()
        self.headless = headless

        self.elevations = self.topo_layer.data.squeeze()
        self.fuels = self.fuel_layer.data.squeeze()

        if self.headless:
            self.image = None
            self.rect = None
        else:
            self.fuel_image = self.fuel_layer._make_pygame_image()
            self.topo_image = self.topo_layer._make_pygame_image()
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
        if not self.headless:
            # This method will update self.image in-place with arr
            arr = pygame.surfarray.pixels3d(self.image)
            arr[burned_idxs[::-1]] = BURNED_RGB_COLOR

    # def _make_terrain_image(self) -> Tuple[pygame.Surface, np.ndarray]:
    #     '''
    #     Create a terrain image representing dryness using the fuel data.
    #     This starts as a numpy array, but is then converted to a pygame.Surface for
    #     compatibility with PyGame.

    #     Returns:
    #         out_surf: The pygame.Surface of the stitched together terrain
    #                   tiles and contour lines
    #     '''
    #     image = np.zeros(self.screen_size + (3, ))

    #     # Loop over the high-level tiles (these are not at the pixel level)
    #     for i in range(self.fuels.shape[0]):
    #         for j in range(self.fuels.shape[1]):
    #             # Need these pixel level coordinates to span the correct range
    #             updated_texture = self._update_texture_dryness(self.fuels[i][i])
    #             image[i, j] = updated_texture

    #     cont_image = self._make_contour_image(image)
    #     out_surf = pygame.surfarray.make_surface(cont_image.swapaxes(0, 1))

    #     return out_surf

    def _make_contour_image(self, image: np.ndarray) -> np.ndarray:
        '''
        Use the image and TopographyLayer to create the elevations array and
        compute the contours. The contours are computed with plt.contours, and the
        contour lines are drawn by converting image to a PIL.Image.Image and using
        the ImageDraw module.

        Arguments:
            image: A numpy array representing the np.float RGB terrain image for display

        Returns:
            out_image: The input image with the contour lines drawn on it
        '''
        # Create a figure with axes
        fig, ax = plt.subplots()
        ax.imshow(image.astype(np.uint8))
        CS = ax.contour(self.elevations, origin='upper')
        ax.clabel(CS, CS.levels, inline=True, fmt=lambda x: f'{x:.0f}')
        plt.axis('off')
        with tempfile.NamedTemporaryFile(suffix='.png') as out_img_path:
            fig.savefig(out_img_path.name, bbox_inches='tight', pad_inches=0)
            out_img = Image.open(out_img_path.name).resize(image.shape[:2])
            # Slice the alpha channel off
            out_img = np.array(out_img)[..., :3]
        plt.close(fig)
        return out_img

    # def _update_texture_dryness(self, fuel: Fuel) -> np.ndarray:
    #     '''
    #     Determine the percent change to make the terrain look drier (i.e.
    #     more red/yellow/brown) by using the FuelArray values. Then, update
    #     the texture color using PIL and image blending with a preset
    #     yellow-brown color/image.

    #     Arguments:
    #         fuel: The Fuel with parameters that specify how "dry" the texture
    #               should look

    #     Returns:
    #         new_texture: The texture with RGB calues modified to look drier based
    #                      on the parameters of fuel_arr
    #     '''
    #     # Add the numbers after normalization
    #     # M_x is inverted because a lower value is more flammable
    #     color_change_pct = fuel.w_0 / 0.2296 + \
    #                        fuel.delta / 7 + \
    #                        (0.2 - fuel.M_x) / 0.2
    #     # Divide by 3 since there are 3 values
    #     color_change_pct /= 3

    #     arr = self.texture.copy()
    #     arr_img = Image.fromarray(arr)
    #     resized_brown = DRY_TERRAIN_BROWN_IMG.resize(arr_img.size)
    #     texture_img = Image.blend(arr_img, resized_brown, color_change_pct / 2)
    #     new_texture = np.array(texture_img)

    #     return new_texture

    # def _make_terrain_layer(self) -> Tuple[pygame.Surface, np.ndarray]:
    #     '''
    #         Load fuel layer RGB values and stack topographic contours

    #         Arguments:
    #             image: A numpy array representing the np.float terrain image

    #         Returns:
    #             out_image: The input image with the RGB values for Fuel Model values
    #     '''

    #     cont_image = self._make_contour_image(self.fuels)
    #     out_surf = pygame.surfarray.make_surface(cont_image.swapaxes(0, 1))

    #     return out_surf


class Fire(pygame.sprite.Sprite):
    '''
    This sprite represents a fire burning on one pixel of the terrain. Its
    image is generally kept very small to make rendering easier. All fire
    spreading is handled by the FireManager it is attached to.
    '''
    def __init__(self, pos: Tuple[int, int], size: int, headless: bool = False) -> None:
        '''
        Initialize the class by recording the position and size of the sprite
        and creating a solid color texture.

        Arguments:
            pos: The (x, y) pixel position of the sprite
            size: The pixel size of the sprite
            headless: Flag to run in a headless state. This will allow PyGame objects to
                      not be initialized.
        '''
        super().__init__()

        self.pos = pos
        self.size = size
        self.headless = headless

        if self.headless:
            self.image = None
            # Need to use self.rect to track the location of the sprite
            # When running headless, we need this to be a tuple instead of a PyGame Rect
            self.rect = pos + (size, size)
        else:
            fire_color = np.zeros((self.size, self.size, 3))
            fire_color[:, :, 0] = 255
            fire_color[:, :, 1] = 153
            fire_color[:, :, 2] = 51
            self.image = pygame.surfarray.make_surface(fire_color)

            self.rect = self.image.get_rect()
            self.rect = self.rect.move(self.pos[0], self.pos[1])

        # Initialize groups to None to start with a "clean" sprite
        self.groups = None

        # Layer 3 so that it appears on top of the terrain and line (if applicable)
        self.layer: int = SpriteLayer.FIRE

    def update(self) -> None:
        '''
        Currently unused.

        '''
        pass


class FireLine(pygame.sprite.Sprite):
    '''
    This sprite represents a fireline on one pixel of the terrain. Its image is generally
    kept very small to make rendering easier. All fireline placement spreading is handled
    by the FireLineManager it is attached to.
    '''
    def __init__(self, pos: Tuple[int, int], size: int, headless: bool = False) -> None:
        '''
        Initialize the class by recording the position and size of the sprite
        and creating a solid color texture.

        Arguments:
            pos: The (x, y) pixel position of the sprite
            size: The pixel size of the sprite
            headless: Flag to run in a headless state. This will allow PyGame objects to
                      not be initialized.

        '''
        super().__init__()

        self.pos = pos
        self.size = size
        self.headless = headless

        if self.headless:
            self.image = None
            # Need to use self.rect to track the location of the sprite
            # When running headless, we need this to be a tuple instead of a PyGame Rect
            self.rect = pos + (size, size)
        else:
            fireline_color = np.zeros((self.size, self.size, 3))
            fireline_color[:, :, 0] = 155  # R
            fireline_color[:, :, 1] = 118  # G
            fireline_color[:, :, 2] = 83  # B
            self.image = pygame.surfarray.make_surface(fireline_color)

            self.rect = self.image.get_rect()
            self.rect = self.rect.move(self.pos[0], self.pos[1])

        # Layer LINE so that it appears on top of the terrain
        self.layer: int = SpriteLayer.LINE

    def update(self) -> None:
        '''
        This doesn't require to be updated right now. May change in the future if we
        learn new things about the physics.

        '''
        pass


class ScratchLine(pygame.sprite.Sprite):
    '''
    This sprite represents a scratch line on one pixel of the terrain. Its image is
    generally kept very small to make rendering easier. All scratch line placement
    spreading is handled by the ScratchLineManager it is attached to.
    '''
    def __init__(self, pos: Tuple[int, int], size: int, headless: bool = False) -> None:
        '''
        Initialize the class by recording the position and size of the sprite
        and creating a solid color texture.
        '''
        super().__init__()

        self.pos = pos
        self.size = size
        self.headless = headless

        if self.headless:
            self.image = None
            # Need to use self.rect to track the location of the sprite
            # When running headless, we need this to be a tuple instead of a PyGame Rect
            self.rect = pos + (size, size)
        else:
            scratchline_color = np.zeros((self.size, self.size, 3))
            scratchline_color[:, :, 0] = 139  # R
            scratchline_color[:, :, 1] = 125  # G
            scratchline_color[:, :, 2] = 58  # B
            self.image = pygame.surfarray.make_surface(scratchline_color)

            self.rect = self.image.get_rect()
            self.rect = self.rect.move(self.pos[0], self.pos[1])

        # Layer LINE so that it appears on top of the terrain
        self.layer: int = SpriteLayer.LINE

    def update(self) -> None:
        '''
        This doesn't require to be updated right now. May change in the future if we
        learn new things about the physics.

        '''
        pass


class WetLine(pygame.sprite.Sprite):
    '''
    This sprite represents a wet line on one pixel of the terrain. Its image is
    generally kept very small to make rendering easier. All wet line placement
    spreading is handled by the WaterLineManager it is attached to.
    '''
    def __init__(self, pos: Tuple[int, int], size: int, headless: bool = False) -> None:
        '''
        Initialize the class by recording the position and size of the sprite
        and creating a color texture.
        '''
        super().__init__()

        self.pos = pos
        self.size = size
        self.headless = headless

        if self.headless:
            self.image = None
            # Need to use self.rect to track the location of the sprite
            # When running headless, we need this to be a tuple instead of a PyGame Rect
            self.rect = pos + (size, size)
        else:
            wetline_color = np.zeros((self.size, self.size, 3))
            wetline_color[:, :, 0] = 212  # R
            wetline_color[:, :, 1] = 241  # G
            wetline_color[:, :, 2] = 249  # B
            self.image = pygame.surfarray.make_surface(wetline_color)

            self.rect = self.image.get_rect()
            self.rect = self.rect.move(self.pos[0], self.pos[1])

        # Layer LINE so that it appears on top of the terrain
        self.layer: int = SpriteLayer.LINE

    def update(self) -> None:
        '''
        This doesn't require to be updated right now. May change in the future if we
        learn new things about the physics.

        '''
        pass
