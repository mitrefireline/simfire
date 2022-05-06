import numpy as np
import pygame
from PIL import Image


def load_image(name: str) -> pygame.surface.Surface:
    """
    Load an image as a pygame.Surface.

    Arguments:
        name: The name/path to the image

    Returns:
        surf: The image loaded as a pygame.Surface
    """
    image = Image.open(name)
    image = np.array(image, dtype=np.uint8)
    # The surfarray creation doesn't work for RGBA arrays, so we need to check the format
    if image.shape[2] < 4:
        surf = pygame.surfarray.make_surface(image)
    else:
        surf = make_surface_rgba(image)

    return surf


def make_surface_rgba(array):
    """Returns a surface made from a [w, h, 4] numpy array with per-pixel alpha"""
    shape = array.shape
    if len(shape) != 3 and shape[2] != 4:
        raise ValueError("Array not RGBA")

    # Create a surface the same width and height as array and with per-pixel alpha
    surface = pygame.Surface(shape[0:2], pygame.SRCALPHA, 32)

    # Copy the rgb part of array to the new surface.
    pygame.pixelcopy.array_to_surface(surface, array[:, :, 0:3])

    # Copy the alpha part of array to the surface using a pixels-alpha view of the surface
    surface_alpha = np.array(surface.get_view("A"), dtype=np.uint8, copy=False)
    surface_alpha[:, :] = array[:, :, 3]

    return surface
