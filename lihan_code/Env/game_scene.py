import random
import os
from typing import Tuple

import pygame
import numpy as np

from Env.constants import *
from Env.spaceship import Spaceship
from Env.obstacle import Obstacle


# os.environ["SDL_VIDEODRIVER"] = "dummy"

class GameScene(object):
    """
    Our shooter game wrapped in a class.
    YELLOW is the AI player. RED is the enemy.
    Methods that start with upper case will be used by Env.
    """

    def __init__(self):
        pygame.font.init()
        # pygame.mixer.init()

        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption(TITLE)

        # self.bullet_hit_sound = pygame.mixer.Sound(HIT_SOUND_PATH)
        # self.bullet_fire_sound = pygame.mixer.Sound(FIRE_SOUND_PATH)

        self.health_font = pygame.font.SysFont(HEALTH_FONT[0], HEALTH_FONT[1])
        self.winner_font = pygame.font.SysFont(WINNER_FONT[0], WINNER_FONT[1])

        yellow_spaceship_image = pygame.transform.scale(pygame.image.load(YELLOW_SPACESHIP_IMAGE_PATH),
                                                        (SPACESHIP_WIDTH, SPACESHIP_HEIGHT))
        red_spaceship_image = pygame.transform.rotate(
            pygame.transform.scale(pygame.image.load(RED_SPACESHIP_IMAGE_PATH), (SPACESHIP_WIDTH, SPACESHIP_HEIGHT)),
            180)

        yellow_shielded_image = pygame.transform.scale(pygame.image.load(YELLOW_SPACESHIP_SHIELDED_IMAGE_PATH),
                                                       (SHIELD_WIDTH, SHIELD_HEIGHT))
        red_shielded_image = pygame.transform.rotate(
            pygame.transform.scale(pygame.image.load(RED_SPACESHIP_SHIELDED_IMAGE_PATH), (SHIELD_WIDTH, SHIELD_HEIGHT)),
            180)

        self.obstacle_image = pygame.transform.scale(pygame.image.load(OBSTACLE_IMAGE_PATH),
                                                     (OBSTACLE_WIDTH, OBSTACLE_HEIGHT))

        if PURE_COLOR_DISPLAY:
            self.background = pygame.Surface((WIDTH, HEIGHT)).convert()
        else:
            # self.background = pygame.transform.scale(pygame.image.load(SPACE_IMAGE_PATH), (WIDTH, HEIGHT))
            self.background = pygame.Surface((WIDTH, HEIGHT)).convert()

        self.player = Spaceship(
            image=yellow_spaceship_image,
            screen_rect=self.screen.get_rect(),
            shielded_image=yellow_shielded_image,
            start_health=YELLOW_START_HEALTH,
            start_x=YELLOW_START_POSITION[0],
            start_y=YELLOW_START_POSITION[1],
            color=YELLOW_COLOR,
            up_direction=True
        )
        self.player_group = pygame.sprite.Group()
        self.player_group.add(self.player)

        self.enemy = Spaceship(
            image=red_spaceship_image,
            shielded_image=red_shielded_image,
            screen_rect=self.screen.get_rect(),
            start_health=RED_START_HEALTH,
            start_x=RED_START_POSITION[0],
            start_y=RED_START_POSITION[1],
            color=RED_COLOR,
            up_direction=False
        )
        self.enemy_group = pygame.sprite.Group()
        self.enemy_group.add(self.enemy)

        self.obstacle_group = pygame.sprite.Group()

        self.clock = pygame.time.Clock()
        self.done = False
        self.reward = 0
        self.enemy_direction = 'left'
        self.Reset()

    def ActionCount(self):
        return len(Action)

    def ScreenShot(self):
        pixels_arr = pygame.surfarray.array3d(self.screen)
        return pixels_arr

    def Done(self):
        return self.done

    def Reward(self):
        return self.reward

    def Reset(self):
        self.player.reset()
        self.enemy.reset()
        self.spawn_obstacles()
        self.done = False

    def Play(self, player_action_num: int):
        # Human input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        if self.done:
            return True

        if player_action_num == -1:
            keys_pressed = pygame.key.get_pressed()
            if keys_pressed[pygame.K_LEFT]:
                player_action_num = 1
            if keys_pressed[pygame.K_RIGHT]:
                player_action_num = 2
            if keys_pressed[pygame.K_UP]:
                player_action_num = 3
            if keys_pressed[pygame.K_DOWN]:
                player_action_num = 4
            if keys_pressed[pygame.K_SPACE]:
                player_action_num = 5
            if keys_pressed[pygame.K_0]:
                player_action_num = 6

        if player_action_num == -1:
            player_action_num = 0

        # Check if game is over
        winner_text = ""
        if self.enemy.is_dead():
            winner_text = "Yellow Wins!"
        elif self.player.is_dead():
            winner_text = "Red Wins!"

        if winner_text != "":
            self.draw_winner(winner_text)
            self.done = True
            return True

        self.reward = 0
        self.update(player_action_num)
        self.draw_window()
        self.clock.tick(FPS)

        return False

    def AdditionalState(self) -> Tuple[int, int, int, int]:
        """
        Returns additional state parameters
        """
        return (
            self.player.health,
            self.player.get_shield_availability(),
            self.enemy.health,
            self.enemy.get_shield_availability()
        )

    def Exit(self):
        pygame.quit()

    # ------------------------- Display update methods -------------------------

    def spawn_obstacles(self):
        """
        Spawns obstacles randomly
        """
        self.obstacle_group.empty()
        for i in range(OBSTACLE_COUNT):
            obstacle = Obstacle(
                image=self.obstacle_image,
                x=random.randrange(0, WIDTH - OBSTACLE_WIDTH, OBSTACLE_WIDTH // 3),
                y=random.randrange(OBSTACLE_Y_MIN, OBSTACLE_Y_MAX, OBSTACLE_HEIGHT // 3)
            )
            self.obstacle_group.add(obstacle)

    def calculate_enemy_action(self):
        """
        Pre-scripted behavior of enemy
        """
        if self.enemy_direction == 'right':
            if self.enemy.rect.left <= 0:
                self.enemy_direction = 'left'
        if self.enemy_direction == 'left':
            if self.enemy.rect.right >= WIDTH:
                self.enemy_direction = 'right'
        fire_or_shield = Action.FIRE if self.enemy.get_shield_availability() > 0 else Action.ACTIVATE_SHIELD
        left_or_right = Action.LEFT if self.enemy_direction == 'left' else Action.RIGHT
        if self.enemy.rect.y < 50:
            up_or_down = Action.UP if random.random() < 0.9 else Action.DOWN
        elif self.enemy.rect.y > OBSTACLE_Y_MIN:
            up_or_down = Action.DOWN if random.random() < 0.9 else Action.UP
        else:
            up_or_down = Action.UP if random.random() < 0.5 else Action.DOWN
        movement = left_or_right if random.random() < 0.7 else up_or_down
        enemy_action = movement if random.random() < 0.7 else fire_or_shield

        return enemy_action

    def update(self, player_action_num: int):
        # Player action from input
        player_action = Action(player_action_num)
        self.player.update(player_action, self.obstacle_group.sprites() + [self.enemy])
        self.player.bullets.update()

        # Enemy action is calculated
        enemy_action = self.calculate_enemy_action()

        self.enemy.update(enemy_action, self.obstacle_group.sprites() + [self.player])
        self.enemy.bullets.update()

        # Check collisions:
        # 1) Player vs enemy bullets
        hit_list = pygame.sprite.spritecollide(self.player, self.enemy.bullets, True)
        if not self.player.shield_activated:
            self.player.health -= BULLET_DAMAGE * len(hit_list)
            self.reward += Reward.BULLET_HIT_PLAYER.value * len(hit_list)

        # 2) Enemy vs player bullets
        hit_list = pygame.sprite.spritecollide(self.enemy, self.player.bullets, True)
        if not self.enemy.shield_activated:
            self.enemy.health -= BULLET_DAMAGE * len(hit_list)
            self.reward += Reward.BULLET_HIT_ENEMY.value * len(hit_list)

        # 3) Bullets vs obstacles
        pygame.sprite.groupcollide(self.obstacle_group, self.player.bullets, False, True)
        pygame.sprite.groupcollide(self.obstacle_group, self.enemy.bullets, False, True)

        if NEGATIVE_REWARD_ENABLED:
            self.reward -= NEGATIVE_REWARD

    def draw_window(self):
        self.screen.blit(self.background, (0, 0))

        # Display texts
        red_health_text = self.health_font.render(
            "Enemy Health: " + str(self.enemy.health), True, WHITE_COLOR)
        yellow_health_text = self.health_font.render(
            "Player Health: " + str(self.player.health), True, WHITE_COLOR)
        self.screen.blit(red_health_text, (WIDTH - red_health_text.get_width() - 10, 10))
        self.screen.blit(yellow_health_text, (10, 10))

        # Draw player
        self.player_group.draw(self.screen)
        self.player.bullets.draw(self.screen)

        # Draw enemy
        self.enemy_group.draw(self.screen)
        self.enemy.bullets.draw(self.screen)

        # Draw obstacles
        self.obstacle_group.draw(self.screen)

        pygame.display.update()

    def draw_winner(self, text):
        draw_text = self.winner_font.render(text, True, WHITE_COLOR)
        self.screen.blit(draw_text, (WIDTH / 2 - draw_text.get_width() /
                                     2, HEIGHT / 2 - draw_text.get_height() / 2))
        pygame.display.update()


if __name__ == "__main__":
    game = GameScene()

    action_list = [1, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 2, 2, 2, 3, 2, 2, 1, 2, 1, 2, 2, 3, 2, 1, 3, 1, 3]

    # for action in action_list:
    #     if game.Done():
    #         break
    #     game.Play(action)
    #     game.Play(action)
    #     game.Play(action)
    #     game.Play(action)
    #     game.Play(action)

    while not game.Done():
        game.Play(-1)

    game.Exit()
