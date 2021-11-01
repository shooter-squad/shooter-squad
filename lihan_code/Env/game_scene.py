import math
import random
from typing import Tuple

import pygame
import numpy as np

from Env.constants import *
from Env.health_pack import HealthPack
from Env.obstacle import Obstacle
from Env.spaceship import Spaceship
from Env.ultimate_ability import UltimateAbility


# os.environ["SDL_VIDEODRIVER"] = "dummy"


def calculate_distance(first: pygame.sprite.Sprite, second: pygame.sprite.Sprite) -> float:
    """
    Calculates the distance between two sprite using their center positions.
    """
    dist = math.sqrt((first.rect.centerx - second.rect.centerx) ** 2 + (first.rect.centery - second.rect.centery) ** 2)
    return dist


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

        # Load images
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
        self.health_pack_image = pygame.transform.scale(pygame.image.load(HEALTH_PACK_IMAGE_PATH),
                                                        (HEALTH_PACK_WIDTH, HEALTH_PACK_HEIGHT))

        yellow_ultimate_ability_image = pygame.transform.scale(pygame.image.load(YELLOW_ULTIMATE_ABILITY_IMAGE_PATH),
                                                               (ULTIMATE_ABILITY_WIDTH, ULTIMATE_ABILITY_HEIGHT))
        red_ultimate_ability_image = pygame.transform.scale(pygame.image.load(RED_ULTIMATE_ABILITY_IMAGE_PATH),
                                                            (ULTIMATE_ABILITY_WIDTH, ULTIMATE_ABILITY_HEIGHT))

        if PURE_COLOR_DISPLAY:
            self.background = pygame.Surface((WIDTH, HEIGHT)).convert()
        else:
            # self.background = pygame.transform.scale(pygame.image.load(SPACE_IMAGE_PATH), (WIDTH, HEIGHT))
            self.background = pygame.Surface((WIDTH, HEIGHT)).convert()

        self.player = Spaceship(
            image=yellow_spaceship_image,
            screen_rect=self.screen.get_rect(),
            shielded_image=yellow_shielded_image,
            ultimate_ability_image=yellow_ultimate_ability_image,
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
            ultimate_ability_image=red_ultimate_ability_image,
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
        self.health_pack_group = pygame.sprite.Group()

        self.clock = pygame.time.Clock()
        self.done = False
        self.reward = 0
        self.enemy_direction = 'left'
        self.frame_count = 0

        self.Reset()

    # ------------------------- Env wrapper methods -------------------------

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
        self.health_pack_group.empty()

        self.reward = 0
        self.enemy_direction = 'left' if random.random() < 0.5 else 'right'
        self.frame_count = 0
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
            if keys_pressed[pygame.K_q]:
                player_action_num = 6
            if keys_pressed[pygame.K_w]:
                player_action_num = 7

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

    def AdditionalState(self) -> np.ndarray:
        """
        Returns additional state parameters
        """
        return np.array([
            self.player.health,
            self.player.get_shield_cool_down(),
            int(self.player.ultimate_available),
            self.enemy.health,
            self.enemy.get_shield_cool_down(),
            int(self.enemy.ultimate_available)
        ])

    def Exit(self):
        pygame.quit()

    # ------------------------- Game logic methods -------------------------

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

    def spawn_health_pack(self):
        """
        Spawn health packs randomly
        """
        self.health_pack_group.empty()
        health_pack = HealthPack(
            image=self.health_pack_image,
            x=random.randrange(0, WIDTH - HEALTH_PACK_WIDTH, HEALTH_PACK_WIDTH // 3),
            y=random.randrange(OBSTACLE_Y_MAX, HEIGHT, HEALTH_PACK_HEIGHT // 3)
        )
        self.health_pack_group.add(health_pack)

    def calculate_enemy_action(self) -> Action:
        """
        Pre-scripted behavior of enemy
        """
        if self.enemy.ultimate_available and calculate_distance(self.enemy, self.player) <= ULTIMATE_ABILITY_WIDTH / 2:
            return Action.USE_ULTIMATE_ABILITY

        if self.enemy_direction == 'right':
            if self.enemy.rect.left <= 0:
                self.enemy_direction = 'left'
        if self.enemy_direction == 'left':
            if self.enemy.rect.right >= WIDTH:
                self.enemy_direction = 'right'
        fire_or_shield = Action.FIRE if self.enemy.get_shield_cool_down() > 0 else Action.ACTIVATE_SHIELD
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
        self.player.ultimate_abilities.update()

        # Enemy action is calculated
        enemy_action = self.calculate_enemy_action()

        self.enemy.update(enemy_action, self.obstacle_group.sprites() + [self.player])
        self.enemy.bullets.update()
        self.enemy.ultimate_abilities.update()

        # Spawn health pack if time is reached
        if self.frame_count == HEALTH_PACK_TIME_INTERVAL:
            self.spawn_health_pack()

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

        # 4) Player vs health pack
        hit_list = pygame.sprite.spritecollide(self.player, self.health_pack_group, True)
        self.player.health += HEALTH_PACK_HEALTH_RECOVERED * len(hit_list)
        self.reward += Reward.PLAYER_GET_HEALTH_PACK.value * len(hit_list)

        # 5) Player vs enemy ultimate
        if not self.player.shield_activated:
            hit_list = pygame.sprite.spritecollide(self.player, self.enemy.ultimate_abilities, False,
                                                   pygame.sprite.collide_mask)
            for hit in hit_list:
                if isinstance(hit, UltimateAbility):
                    damage = hit.get_damage()
                    self.player.health -= damage
                    if damage > 0:
                        self.reward += Reward.ULTIMATE_HIT_PLAYER.value

        # 6) Enemy vs player ultimate
        if not self.enemy.shield_activated:
            hit_list = pygame.sprite.spritecollide(self.enemy, self.player.ultimate_abilities, False,
                                                   pygame.sprite.collide_mask)
            for hit in hit_list:
                if isinstance(hit, UltimateAbility):
                    damage = hit.get_damage()
                    self.enemy.health -= damage
                    if damage > 0:
                        self.reward += Reward.ULTIMATE_HIT_ENEMY.value

        if NEGATIVE_REWARD_ENABLED:
            self.reward -= NEGATIVE_REWARD

    def draw_window(self):
        self.screen.blit(self.background, (0, 0))

        # Draw player
        self.player.ultimate_abilities.draw(self.screen)
        self.player_group.draw(self.screen)
        self.player.bullets.draw(self.screen)

        # Draw enemy
        self.enemy.ultimate_abilities.draw(self.screen)
        self.enemy_group.draw(self.screen)
        self.enemy.bullets.draw(self.screen)

        # Draw obstacles and health packs
        self.obstacle_group.draw(self.screen)
        self.health_pack_group.draw(self.screen)

        # Display texts
        red_health_text = self.health_font.render(
            "Enemy Health: " + str(self.enemy.health), True, WHITE_COLOR)
        yellow_health_text = self.health_font.render(
            "Player Health: " + str(self.player.health), True, WHITE_COLOR)
        self.screen.blit(red_health_text, (WIDTH - red_health_text.get_width() - 10, 10))
        self.screen.blit(yellow_health_text, (10, 10))

        pygame.display.update()
        self.frame_count += 1

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
