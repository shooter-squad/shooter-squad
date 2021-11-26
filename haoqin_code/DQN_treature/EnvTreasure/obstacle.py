import pygame


class Obstacle(pygame.sprite.Sprite):
    """
    The obstacle class.
    """

    def __init__(self, image: pygame.Surface, x: int, y: int):
        super().__init__()

        self.image = image
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
