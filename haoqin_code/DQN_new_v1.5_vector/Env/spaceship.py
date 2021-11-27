from typing import List, Tuple
from enum import Enum

import pygame

from Env.bullet import Bullet
from Env.ultimate_ability import UltimateAbility
from Env.constants import *


class SpaceshipType(Enum):
    """
    All enemy types in the game.
    """
    NORMAL_ENEMY = 0
    CHARGE_ENEMY = 1
    PLAYER = 2


class Spaceship(pygame.sprite.Sprite):
    """
    The spaceship class for both player and enemy.
    """

    def __init__(self, image: pygame.Surface, shielded_image: pygame.Surface, ultimate_ability_image: pygame.Surface,
                 screen_rect: pygame.Rect, start_health: int, start_x: int, start_y: int, color: Tuple[int, int, int],
                 up_direction: bool, is_player: bool, type: SpaceshipType = SpaceshipType.NORMAL_ENEMY):
        super().__init__()

        if PURE_COLOR_DISPLAY:
            self.image = pygame.Surface((SPACESHIP_WIDTH, SPACESHIP_HEIGHT)).convert()
            self.image.fill(color)
        else:
            self.image = image

        self.original_image = image
        self.shielded_image = shielded_image

        self.rect = self.image.get_rect()
        self.rect.x = start_x
        self.rect.y = start_y
        self.screen_rect = screen_rect
        self.health = start_health
        self.start_health = start_health
        self.start_x = start_x
        self.start_y = start_y
        self.color = color
        self.up_direction = up_direction
        self.type = type

        self.bullets = pygame.sprite.Group()
        self.bullet_interval = PLAYER_BULLET_INTERVAL if is_player else ENEMY_BULLET_INTERVAL
        self.time_since_last_bullet = self.bullet_interval

        self.shield_activated = False
        self.time_since_shield_activated = SHIELD_COOL_DOWN
        self.shield_enabled = ENEMY_SHIELD_ENABLED or is_player

        self.ultimate_ability_image = ultimate_ability_image
        self.ultimate_abilities = pygame.sprite.Group()
        self.ultimate_available = True

        self.type = type
        self.enemy_behavior = Action.NOOP  # Only used for enemies

    def update(self, action: Action, others: List[pygame.sprite.Sprite]):
        # Update counters
        self.time_since_shield_activated += 1
        self.time_since_last_bullet += 1

        if self.shield_activated and self.time_since_shield_activated >= SHIELD_DURATION:
            self.deactivate_shield()

        if action == Action.NOOP:
            return

        if action == Action.FIRE:
            self.fire()
        # TODO: disabled UP and DOWN
        # elif action == Action.ACTIVATE_SHIELD:
        #     self.activate_shield()
        # elif action == Action.USE_ULTIMATE_ABILITY:
        #     self.use_ultimate_ability()
        else:
            vel = 1 if self.up_direction else -1
            if action == Action.LEFT:
                vel *= -VEL
            elif action == Action.RIGHT:
                vel *= VEL
            # elif action == Action.UP:
            #     vel *= -VEL
            # elif action == Action.DOWN:
            #     vel *= VEL

            if action in [Action.LEFT, Action.RIGHT]:
                self.rect.x += vel
            # elif action in [Action.UP, Action.DOWN]:
            #     self.rect.y += vel

            for other in others:
                if self.type == SpaceshipType.CHARGE_ENEMY:
                    continue
                if isinstance(other, Spaceship) and other.is_dead():
                    continue
                if pygame.sprite.collide_rect(other, self):
                    if action in [Action.LEFT, Action.RIGHT]:
                        self.rect.x -= vel
                    # elif action in [Action.UP, Action.DOWN]:
                    #     self.rect.y -= vel
                    break

        # Keep sprite on screen
        self.rect.clamp_ip(self.screen_rect)

    def draw(self, screen: pygame.Surface):
        screen.blit(self.image, self.rect)

    def fire(self):
        if self.time_since_last_bullet < self.bullet_interval:
            return

        self.bullets.add(
            Bullet(
                centerx=self.rect.centerx,
                centery=self.rect.y if self.up_direction else self.rect.bottom,
                color=self.color,
                vel_x=0,
                vel_y=-BULLET_VEL if self.up_direction else BULLET_VEL,
                screen_rect=self.screen_rect
            )
        )
        self.time_since_last_bullet = 0

    def get_shield_cool_down(self) -> int:
        return max(0, SHIELD_COOL_DOWN - self.time_since_shield_activated)

    def activate_shield(self):
        if not self.shield_enabled or self.time_since_shield_activated < SHIELD_COOL_DOWN:
            return

        old_x = self.rect.centerx
        old_y = self.rect.centery

        self.image = self.shielded_image
        self.rect = self.image.get_rect()
        self.rect.centerx = old_x
        self.rect.centery = old_y
        self.shield_activated = True

        self.time_since_shield_activated = 0

    def deactivate_shield(self):
        if not self.shield_enabled:
            return
        old_x = self.rect.centerx
        old_y = self.rect.centery

        self.image = self.original_image
        self.rect = self.image.get_rect()
        self.rect.centerx = old_x
        self.rect.centery = old_y
        self.shield_activated = False

    def use_ultimate_ability(self):
        if not self.ultimate_available:
            return

        ultimate_ability = UltimateAbility(
            image=self.ultimate_ability_image,
            centerx=self.rect.centerx,
            centery=self.rect.centery
        )
        self.ultimate_abilities.add(ultimate_ability)
        self.ultimate_available = False

    def reset(self):
        self.rect.x = self.start_x
        self.rect.y = self.start_y
        self.health = self.start_health

        self.bullets.empty()
        self.time_since_last_bullet = self.bullet_interval

        self.deactivate_shield()
        self.time_since_shield_activated = SHIELD_COOL_DOWN

        self.ultimate_abilities.empty()
        self.ultimate_available = True

        self.enemy_behavior = Action.NOOP

    def is_dead(self) -> bool:
        return self.health <= 0
