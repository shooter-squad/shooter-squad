import pygame
import math

from .constants import *


class UltimateAbility(pygame.sprite.Sprite):
    """
    The ultimate ability class.
    """

    def __init__(self, image: pygame.Surface, centerx: int, centery: int):
        super().__init__()

        self.image = image
        self.rect = self.image.get_rect()
        self.rect.centerx = centerx
        self.rect.centery = centery
        self.mask = pygame.mask.from_surface(self.image)
        self.alpha = 255
        self.frame_counter = 0

    def update(self):
        self.alpha = max(0, self.alpha - 5)
        if self.alpha == 0:
            self.kill()

        self.image.set_alpha(self.alpha)
        self.frame_counter += 1

    def get_damage(self):
        if self.frame_counter % ULTIMATE_ABILITY_DAMAGE_INTERVAL != 1:
            return 0
        return math.ceil(ULTIMATE_ABILITY_MAX_DAMAGE * self.alpha / 255)
