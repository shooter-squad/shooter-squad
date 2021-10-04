from typing import List, Tuple

import pygame

from .bullet import Bullet
from .constants import *


class Spaceship(pygame.sprite.Sprite):
    """
    The spaceship class for both player and enemy.
    """

    def __init__(self, image: pygame.Surface, screen_rect: pygame.Rect, start_health: int, start_x: int, start_y: int,
                 color: Tuple[int, int, int], up_direction: bool):
        super().__init__()

        if PURE_COLOR_DISPLAY:
            self.image = pygame.Surface((SPACESHIP_WIDTH, SPACESHIP_HEIGHT)).convert()
            self.image.fill(color)
        else:
            self.image = image
        
        self.rect = self.image.get_rect()
        self.screen_rect = screen_rect
        self.health = start_health
        self.start_health = start_health
        self.start_x = start_x
        self.start_y = start_y
        self.color = color
        self.up_direction = up_direction
        self.bullets = pygame.sprite.Group()

    def update(self, action: Action, others: List[pygame.sprite.Sprite]):
        if action == Action.NOOP:
            return

        if action == Action.FIRE:
            self.fire()
        else:
            vel = 1 if self.up_direction else -1
            if action == Action.LEFT:
                vel *= -VEL
            elif action == Action.RIGHT:
                vel *= VEL

            self.rect.x += vel
            for other in others:
                if pygame.sprite.collide_rect(other, self):
                    self.rect.x -= vel
                    break

        # Keep sprite on screen
        self.rect.clamp_ip(self.screen_rect)

    def fire(self):
        if len(self.bullets) >= MAX_BULLETS:
            return

        self.bullets.add(
            Bullet(
                x=self.rect.centerx,
                y=self.rect.y if self.up_direction else self.rect.bottom,
                color=self.color,
                vel_x=0,
                vel_y=-BULLET_VEL if self.up_direction else BULLET_VEL,
                screen_rect=self.screen_rect
            )
        )

    def reset(self):
        self.rect.x = self.start_x
        self.rect.y = self.start_y
        self.health = self.start_health
        self.bullets.empty()

    def is_dead(self) -> bool:
        return self.health <= 0
