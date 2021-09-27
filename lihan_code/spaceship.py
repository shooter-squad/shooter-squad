from typing import List, Tuple

import pygame

from bullet import Bullet
from constants import *


class Spaceship(pygame.sprite.Sprite):
    """
    The spaceship class for both player and enemy.
    """

    def __init__(self, image: pygame.Surface, screen_rect: pygame.Rect, start_health: int, start_x: int, start_y: int,
                 color: Tuple[int, int, int], up_direction: bool):
        super().__init__()

        self.frame_to_update = 0
        self.image = image
        self.surface = image
        self.direction = 0
        self.rect = self.image.get_rect()
        self.screen_rect = screen_rect
        self.health = start_health
        self.start_health = start_health
        self.start_x = start_x
        self.start_y = start_y
        self.color = color
        self.up_direction = up_direction
        self.bullets = pygame.sprite.Group()
        self.action = Action.NOOP

    def update(self, action: Action, others: List[pygame.sprite.Sprite]):
        if self.frame_to_update > 0:
            self.frame_to_update -= 1
            return
        self.frame_to_update = FRAME_TO_UPDATE

        # Handle actions
        self.action = action
        if action == Action.NOOP:
            return
        if action == Action.FIRE:
            self.fire()
        if action in [Action.LEFT, Action.RIGHT, Action.UP, Action.DOWN]:
            vel = 1 if self.up_direction else -1
            if action == Action.LEFT or action == Action.UP:
                vel *= -VEL
            else:
                vel *= VEL

            if action == Action.LEFT or action == Action.RIGHT:
                self.rect.x += vel
                # Check collision
                for other in others:
                    if pygame.sprite.collide_rect(other, self):
                        self.rect.x -= vel
                        break
            else:
                self.rect.y += vel
                # Check collision
                for other in others:
                    if pygame.sprite.collide_rect(other, self):
                        self.rect.y -= vel
                        break
        if action == Action.TURN_COUNTERCLOCKWISE or action == Action.TURN_CLOCKWISE:
            self.rot_center(THETA * (1 if action == Action.TURN_CLOCKWISE else -1))

        # Keep sprite on screen
        self.rect.clamp_ip(self.screen_rect)

    def rot_center(self, angle):
        """
        Rotate an image while keeping its center
        """
        self.direction += angle
        self.image = pygame.transform.rotate(self.surface, self.direction)

    def fire(self):
        if len(self.bullets) >= MAX_BULLETS:
            return

        self.bullets.add(
            Bullet(
                x=self.rect.centerx,
                y=self.rect.centery,
                color=self.color,
                direction=self.direction,
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
