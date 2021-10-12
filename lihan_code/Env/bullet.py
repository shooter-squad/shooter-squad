from typing import Tuple

import pygame
from constants import *


class Bullet(pygame.sprite.Sprite):
    """
    The bullet class.
    """

    def __init__(self, x: int, y: int, color: Tuple[int, int, int], vel_x: int, vel_y: int, screen_rect: pygame.Rect):
        super().__init__()
        self.image = pygame.Surface((BULLET_WIDTH, BULLET_HEIGHT)).convert()
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.vel_x = vel_x
        self.vel_y = vel_y
        self.screen_rect = screen_rect

    def update(self):
        self.rect.x += self.vel_x
        self.rect.y += self.vel_y

        # Kill if it goes off screen
        if not self.screen_rect.colliderect(self.rect):
            self.kill()
