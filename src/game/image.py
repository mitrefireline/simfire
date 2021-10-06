import numpy as np
import pygame


def load_image(name: str):
    """ Load image and return image object"""
    image = pygame.image.load(name)
    if image.get_alpha() is None:
        image = image.convert()
    else:
        image = image.convert_alpha()
    return image


def set_image_dryness(surface: pygame.Surface, percent_change: float) -> None:
    '''
    Modify a terrain texture to look more yellow depending on its moisture
    content. This is done by directly accesing the pixel values with a
    surfarray and changing the red and green channels to larger values to
    create a more yellow look. 
    '''
    arr = pygame.surfarray.pixels3d(surface)
    arr[...,0] = np.clip((1+percent_change)*arr[...,0], 0, 255).astype(np.uint8)
    arr[...,0] = np.clip(0.75*(1+percent_change)*arr[...,0], 0, 255).astype(np.uint8)
    return
