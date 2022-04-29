import tempfile
from typing import Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pygame
from reportlab.graphics import renderPM
from svglib.svglib import svg2rlg

from ..enums import (BurnStatus, SpriteLayer, BURNED_RGB_COLOR)
from ..utils.layers import FuelLayer, TopographyLayer


class Terrain(pygame.sprite.Sprite):
    '''
    Use a TopographyLayer and a FuelLayer to make terrain. This sprite is just the
    entire background that the fire appears on. This sprite changes the color of each
    terrain pixel based on its "dryness/flammability" if it is unburned, or based on
    whether the pixel is a fireline or burned.
    '''
    def __init__(self,
                 fuel_layer: FuelLayer,
                 topo_layer: TopographyLayer,
                 screen_size: Tuple[int, int],
                 headless: bool = False) -> None:
        '''
        Initialize the class by loading the tile textures and stitching
        together the whole terrain image.

        Arguments:
            fuel_layer: A FuelLayer containing a `data` parameter that is a numpy array
                        containing a Fuel objects
            topo_layer: A TopographyLayer that describes the topography (elevation) of the
                        terrain as a numpy array in it `data` parameter
            screen_size: The game's screen size in pixels
            headless: Flag to run in a headless state. This will allow PyGame objects to
                      not be initialized.

        '''
        super().__init__()

        self.fuel_layer = fuel_layer
        self.topo_layer = topo_layer

        self.screen_size = screen_size
        self.headless = headless

        self.elevations = self.topo_layer.data.squeeze()
        self.fuels = self.fuel_layer.data.squeeze()

        self.image: Optional[pygame.surface.Surface]
        self.rect: Optional[pygame.Rect]

        if not self.headless:
            # Create the terrain image
            terrain_image = self._make_terrain_image()
            # Convert the terrain image to a PyGame surface for display
            self.image = pygame.surfarray.make_surface(terrain_image.swapaxes(0, 1))
            # The rectangle for this sprite is the entire game
            self.rect = pygame.Rect(0, 0, *self.screen_size)
        else:
            self.image = None
            self.rect = None

        # This sprite should always have layer 1 since it will always
        # be behind every other sprite
        self.layer = SpriteLayer.TERRAIN

    def update(self, *args: Any, **kwargs: Any) -> None:
        '''
        Change any burned squares to brown using fire_map, which
        contains pixel-wise values for tile burn status.
        The parent class update() expects just args and kwargs, so we

        Arguments:
            a
            fire_map: A 2-D numpy array containing enumerated values for unburned,
                      burning, and burned tile status for the game screen

        Returns: None
        '''
        # Argument checks for compatibility
        if len(args) == 0 or len(args) > 1:
            raise ValueError('The input arguments to update() should contain'
                             'only one value for the fire_map. Instead got: '
                             f'{args}')
        if len(kwargs) > 0:
            raise ValueError('The input keyword arguments to update() should '
                             f'contain no values. Instead got: {kwargs}')
        if not isinstance(args[0], np.ndarray):
            raise TypeError('The input fire_map should be a numpy array. '
                            f'Instead got: {args[0]}')

        fire_map: np.ndarray = args[0]
        if fire_map.shape != self.screen_size:
            raise ValueError(f'The shape of the fire_map {fire_map.shape} does '
                             f'not match the shape of the screen {self.screen_size}')
        fire_map = fire_map.copy()
        self._update(fire_map)

    def _update(self, fire_map: np.ndarray) -> None:
        '''
        Internal method to update the burned squares to brown. This will update
        self.image in-place.
        This is needed because the parent class Sprite.update() only take in
        args and kwargs, and we want to do input checking on those parameters
        before updating.

        Arguments:
            fire_map: A 2-D numpy array containing enumerated values for unburned,
                      burning, and burned tile status for the game screen
        '''
        burned_idxs = np.where(fire_map == BurnStatus.BURNED)
        if not self.headless:
            # This method will update self.image in-place with arr
            if self.image is not None:
                arr = pygame.surfarray.pixels3d(self.image)
            arr[burned_idxs[::-1]] = BURNED_RGB_COLOR

    def _make_terrain_image(self) -> np.ndarray:
        '''
        Use the FuelLayer image and TopographyLayer contours to create the
        terrain background image. This will show the FuelLayer as the landscape/overhead
        view, with the contour lines overlaid on top.

        Arguments:
            None

        Returns:
            out_image: The input image with the contour lines drawn on it
        '''
        image = self.fuel_layer.image.squeeze()
        # Create a figure with axes
        fig, ax = plt.subplots()
        # The fmt argument will display the levels as whole numbers (otherwise
        # the decimal points look messy)
        contours = ax.contour(self.topo_layer.data.squeeze(),
                              origin='upper',
                              colors='black')
        ax.clabel(contours,
                  contours.levels,
                  inline=True,
                  fmt=lambda x: f'{x:.0f}',
                  fontsize='large')
        ax.imshow(image.astype(np.uint8))
        plt.axis('off')

        # Save the figure as a vector graphic to get just the image (no axes,
        # ticks, figure edges, etc.)
        # Then load it, resize, and convert to numpy
        with tempfile.NamedTemporaryFile(suffix='.svg') as out_img_path:
            fig.savefig(out_img_path.name, bbox_inches='tight', pad_inches=0)
            drawing = svg2rlg(out_img_path.name)
            # Resize the SVG drawing
            scale_x = image.shape[1] / drawing.width
            scale_y = image.shape[0] / drawing.height
            drawing.width = drawing.width * scale_x
            drawing.height = drawing.height * scale_y
            drawing.scale(scale_x, scale_y)
            # Convert to Pillow
            out_img_pil = renderPM.drawToPIL(drawing)
        plt.close(fig)
        # Slice the alpha channel off
        out_img = np.array(out_img_pil)[..., :3]

        return out_img


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
            self.rect = pygame.Rect(*(pos + (size, size)))
        else:
            fire_color = np.zeros((self.size, self.size, 3))
            fire_color[:, :, 0] = 255
            fire_color[:, :, 1] = 153
            fire_color[:, :, 2] = 51
            self.image = pygame.surfarray.make_surface(fire_color)

            self.rect = self.image.get_rect()
            self.rect = self.rect.move(self.pos[0], self.pos[1])

        # Layer 3 so that it appears on top of the terrain and line (if applicable)
        self.layer: int = SpriteLayer.FIRE

    def update(self, *args, **kwargs) -> None:
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
            self.rect = pygame.Rect(*(pos + (size, size)))
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

    def update(self, *args, **kwargs) -> None:
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
            self.rect = pygame.Rect(*(pos + (size, size)))
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

    def update(self, *args, **kwargs) -> None:
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
            self.rect = pygame.Rect(*(pos + (size, size)))
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

    def update(self, *args, **kwargs) -> None:
        '''
        This doesn't require to be updated right now. May change in the future if we
        learn new things about the physics.

        '''
        pass
