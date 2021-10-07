import pygame


def load_image(name: str) -> pygame.Surface:
    '''
    Load an image as a pygame.Surface.

    Arguments:
        name: The name/path to the image

    Returns:
        image: The image loaded as a pygame.Surface
    '''
    image = pygame.image.load(name)
    if image.get_alpha() is None:
        image = image.convert()
    else:
        image = image.convert_alpha()
    return image
