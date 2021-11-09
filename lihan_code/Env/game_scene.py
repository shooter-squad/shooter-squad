import math
import random
from typing import Tuple

import pygame
import numpy as np

from Env.constants import *
from Env.health_pack import HealthPack
from Env.obstacle import Obstacle
from Env.spaceship import Spaceship, SpaceshipType
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
        self.red_spaceship_image = pygame.transform.rotate(
            pygame.transform.scale(pygame.image.load(RED_SPACESHIP_IMAGE_PATH), (SPACESHIP_WIDTH, SPACESHIP_HEIGHT)),
            180)
        self.blue_spaceship_image = pygame.transform.rotate(
            pygame.transform.scale(pygame.image.load(BLUE_SPACESHIP_IMAGE_PATH), (SPACESHIP_WIDTH, SPACESHIP_HEIGHT)),
            180)

        yellow_shielded_image = pygame.transform.scale(pygame.image.load(YELLOW_SPACESHIP_SHIELDED_IMAGE_PATH),
                                                       (SHIELD_WIDTH, SHIELD_HEIGHT))
        self.red_shielded_image = pygame.transform.rotate(
            pygame.transform.scale(pygame.image.load(RED_SPACESHIP_SHIELDED_IMAGE_PATH), (SHIELD_WIDTH, SHIELD_HEIGHT)),
            180)

        self.obstacle_image = pygame.transform.scale(pygame.image.load(OBSTACLE_IMAGE_PATH),
                                                     (OBSTACLE_WIDTH, OBSTACLE_HEIGHT))
        self.health_pack_image = pygame.transform.scale(pygame.image.load(HEALTH_PACK_IMAGE_PATH),
                                                        (HEALTH_PACK_WIDTH, HEALTH_PACK_HEIGHT))

        yellow_ultimate_ability_image = pygame.transform.scale(pygame.image.load(YELLOW_ULTIMATE_ABILITY_IMAGE_PATH),
                                                               (ULTIMATE_ABILITY_WIDTH, ULTIMATE_ABILITY_HEIGHT))
        self.red_ultimate_ability_image = pygame.transform.scale(pygame.image.load(RED_ULTIMATE_ABILITY_IMAGE_PATH),
                                                                 (ULTIMATE_ABILITY_WIDTH, ULTIMATE_ABILITY_HEIGHT))
        self.blue_ultimate_ability_image = pygame.transform.scale(pygame.image.load(BLUE_ULTIMATE_ABILITY_IMAGE_PATH),
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
            up_direction=True,
            is_player=True
        )
        self.player_group = pygame.sprite.Group()
        self.player_group.add(self.player)

        self.enemy_group = pygame.sprite.Group()

        start_x = random.randrange(0, WIDTH - SPACESHIP_WIDTH)
        for i in range(NORMAL_ENEMY_COUNT):
            start_y = random.randrange(ENEMY_START_Y_RANGES[i][0], ENEMY_START_Y_RANGES[i][1])
            enemy = Spaceship(
                image=self.red_spaceship_image,
                shielded_image=self.red_shielded_image,
                ultimate_ability_image=self.red_ultimate_ability_image,
                screen_rect=self.screen.get_rect(),
                start_health=RED_START_HEALTH,
                start_x=start_x,
                start_y=start_y,
                color=RED_COLOR,
                up_direction=False,
                is_player=False
            )
            start_x += WIDTH // NORMAL_ENEMY_COUNT
            enemy.enemy_behavior = Action.LEFT if random.random() < 0.5 else Action.RIGHT
            self.enemy_group.add(enemy)

        self.obstacle_group = pygame.sprite.Group()
        self.health_pack_group = pygame.sprite.Group()

        self.clock = pygame.time.Clock()
        self.done = False
        self.reward = 0
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
        for enemy in self.enemy_group.sprites():
            if isinstance(enemy, Spaceship):
                if enemy.type == SpaceshipType.NORMAL_ENEMY:
                    enemy.reset()
                    enemy.enemy_behavior = Action.LEFT if random.random() < 0.5 else Action.RIGHT
                else:
                    enemy.kill()
        self.spawn_obstacles()
        self.health_pack_group.empty()

        self.reward = 0
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
        all_enemies_dead = True
        for enemy in self.enemy_group.sprites():
            if isinstance(enemy, Spaceship) and not enemy.is_dead():
                all_enemies_dead = False
                break

        if all_enemies_dead:
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
        player_arr = np.zeros(ADDITIONAL_STATE_LEN_PLAYER, dtype=np.int64)
        normal_arr = np.zeros(ADDITIONAL_STATE_LEN_NORMAL, dtype=np.int64)
        charge_arr = np.zeros(ADDITIONAL_STATE_LEN_CHARGE, dtype=np.int64)

        player_temp = np.array([
            self.player.health,
            self.player.get_shield_cool_down(),
            int(self.player.ultimate_available)
        ])
        player_arr[0:player_temp.shape[0]] = player_temp

        normal_temp = []
        charge_temp = []
        for enemy in self.enemy_group.sprites():
            if isinstance(enemy, Spaceship) and enemy.type == SpaceshipType.NORMAL_ENEMY:
                normal_temp += [
                    enemy.health,
                    enemy.get_shield_cool_down(),
                    int(enemy.ultimate_available)
                ]
            elif isinstance(enemy, Spaceship) and enemy.type == SpaceshipType.CHARGE_ENEMY:
                charge_temp += [
                    enemy.health,
                    enemy.get_shield_cool_down(),
                    int(enemy.ultimate_available)
                ]

        normal_temp = np.array(normal_temp)
        charge_temp = np.array(charge_temp)

        normal_arr[0:normal_temp.shape[0]] = normal_temp
        charge_arr[0:charge_temp.shape[0]] = charge_temp

        return np.concatenate((player_arr, normal_arr, charge_arr), axis=None)

    def Exit(self):
        pygame.quit()

    # ------------------------- Game logic methods -------------------------

    def spawn_obstacles(self):
        """
        Spawns obstacles randomly
        """
        self.obstacle_group.empty()
        for i in range(OBSTACLE_COUNT):
            spawn_left = True if random.random() < 0.5 else False
            obstacle = Obstacle(
                image=self.obstacle_image,
                x=random.randrange(0 if spawn_left else WIDTH // 2 + SPACESHIP_WIDTH,
                                   WIDTH // 2 - OBSTACLE_WIDTH - SPACESHIP_WIDTH if spawn_left else WIDTH - OBSTACLE_WIDTH,
                                   OBSTACLE_WIDTH // 3),
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

    def spawn_charge_enemy(self):
        """
        Spawn charge enemy
        """
        enemy = Spaceship(
            image=self.blue_spaceship_image,
            shielded_image=self.red_shielded_image,
            ultimate_ability_image=self.blue_ultimate_ability_image,
            screen_rect=self.screen.get_rect(),
            start_health=RED_START_HEALTH,
            start_x=random.randrange(WIDTH // 2 - 120, WIDTH // 2 + 120 - SPACESHIP_WIDTH),
            start_y=0,
            color=BLUE_COLOR,
            up_direction=False,
            is_player=False,
            type=SpaceshipType.CHARGE_ENEMY
        )
        enemy.enemy_behavior = Action.UP
        self.enemy_group.add(enemy)

    def calculate_enemy_action(self, enemy: Spaceship, i: int) -> Action:
        """
        Pre-scripted behavior of enemy
        """
        if enemy.ultimate_available and calculate_distance(enemy, self.player) <= ULTIMATE_ABILITY_WIDTH / 2:
            return Action.USE_ULTIMATE_ABILITY

        if enemy.enemy_behavior == Action.RIGHT:
            if enemy.rect.left <= 0:
                enemy.enemy_behavior = Action.LEFT
        if enemy.enemy_behavior == Action.LEFT:
            if enemy.rect.right >= WIDTH:
                enemy.enemy_behavior = Action.RIGHT
        fire_or_shield = Action.ACTIVATE_SHIELD if enemy.shield_enabled and enemy.get_shield_cool_down() == 0 else Action.FIRE
        if enemy.type == SpaceshipType.CHARGE_ENEMY or enemy.enemy_behavior == Action.UP:
            if enemy.rect.bottom >= HEIGHT:
                enemy.health = 0  # soft kill
            movement = Action.UP
        else:
            left_or_right = Action.LEFT if enemy.enemy_behavior == Action.LEFT else Action.RIGHT
            if enemy.rect.y < ENEMY_START_Y_RANGES[i][0] + 50:
                up_or_down = Action.UP if random.random() < 0.9 else Action.DOWN
            elif enemy.rect.y > ENEMY_START_Y_RANGES[i][1]:
                up_or_down = Action.DOWN if random.random() < 0.9 else Action.UP
            else:
                up_or_down = Action.UP if random.random() < 0.5 else Action.DOWN
            movement = left_or_right if random.random() < 0.95 else up_or_down
        enemy_action = movement if random.random() < 0.7 else fire_or_shield

        return enemy_action

    def update(self, player_action_num: int):
        # Player action from input
        player_action = Action(player_action_num)
        self.player.update(player_action, self.obstacle_group.sprites() + self.enemy_group.sprites())
        self.player.bullets.update()
        self.player.ultimate_abilities.update()

        # Enemy action is calculated
        i = 0
        for enemy in self.enemy_group.sprites():
            if isinstance(enemy, Spaceship):
                if not enemy.is_dead():
                    enemy_action = self.calculate_enemy_action(enemy, i)
                    enemy.update(enemy_action, self.obstacle_group.sprites() + [self.player])
                enemy.bullets.update()
                enemy.ultimate_abilities.update()
            i += 1

        # Spawn health pack if time is reached
        if self.frame_count == HEALTH_PACK_TIME_INTERVAL:
            self.spawn_health_pack()

        # Spawn health pack if time is reached
        if self.frame_count == CHARGE_ENEMY_SPAWN_INTERVAL:
            self.spawn_charge_enemy()

        # Check collisions:
        for enemy in self.enemy_group.sprites():
            if isinstance(enemy, Spaceship):
                # 1) Player vs enemy bullets
                hit_list = pygame.sprite.spritecollide(self.player, enemy.bullets, True)
                if not self.player.shield_activated:
                    self.player.health -= BULLET_DAMAGE * len(hit_list)
                    self.reward += Reward.BULLET_HIT_PLAYER.value * len(hit_list)

                # 2) Enemy vs player bullets
                if not enemy.is_dead():
                    hit_list = pygame.sprite.spritecollide(enemy, self.player.bullets, True)
                    if not enemy.shield_activated:
                        enemy.health -= BULLET_DAMAGE * len(hit_list)
                        self.reward += Reward.BULLET_HIT_ENEMY.value * len(hit_list)

                # 3) Bullets vs obstacles
                pygame.sprite.groupcollide(self.obstacle_group, enemy.bullets, False, True)

                # 5) Player vs enemy ultimate
                if not self.player.shield_activated:
                    hit_list = pygame.sprite.spritecollide(self.player, enemy.ultimate_abilities, False,
                                                           pygame.sprite.collide_mask)
                    for hit in hit_list:
                        if isinstance(hit, UltimateAbility):
                            damage = hit.get_damage()
                            self.player.health -= damage
                            if damage > 0:
                                self.reward += Reward.ULTIMATE_HIT_PLAYER.value

                # 6) Enemy vs player ultimate
                if not enemy.is_dead() and not enemy.shield_activated:
                    hit_list = pygame.sprite.spritecollide(enemy, self.player.ultimate_abilities, False,
                                                           pygame.sprite.collide_mask)
                    for hit in hit_list:
                        if isinstance(hit, UltimateAbility):
                            damage = hit.get_damage()
                            enemy.health -= damage
                            if damage > 0:
                                self.reward += Reward.ULTIMATE_HIT_ENEMY.value

                # 7) Charge enemy vs player
                if enemy.type == SpaceshipType.CHARGE_ENEMY and \
                        not enemy.is_dead() and \
                        pygame.sprite.collide_rect(enemy, self.player):
                    enemy.health = 0
                    if not self.player.shield_activated:
                        self.player.health = 0
                        self.reward += Reward.PLAYER_HIT_CHARGE_ENEMY.value

        # 3) Bullets vs obstacles
        pygame.sprite.groupcollide(self.obstacle_group, self.player.bullets, False, True)

        # 4) Player vs health pack
        hit_list = pygame.sprite.spritecollide(self.player, self.health_pack_group, True)
        self.player.health += HEALTH_PACK_HEALTH_RECOVERED * len(hit_list)
        self.reward += Reward.PLAYER_GET_HEALTH_PACK.value * len(hit_list)

        if NEGATIVE_REWARD_ENABLED:
            self.reward -= NEGATIVE_REWARD

        print(self.reward)

    def draw_window(self):
        self.screen.blit(self.background, (0, 0))

        # Draw player
        self.player.ultimate_abilities.draw(self.screen)
        self.player_group.draw(self.screen)
        self.player.bullets.draw(self.screen)

        # Draw enemy
        for enemy in self.enemy_group.sprites():
            if isinstance(enemy, Spaceship):
                enemy.ultimate_abilities.draw(self.screen)
                enemy.bullets.draw(self.screen)
                if not enemy.is_dead():
                    enemy.draw(self.screen)

        # self.enemy_group.draw(self.screen)

        # Draw obstacles and health packs
        self.obstacle_group.draw(self.screen)
        self.health_pack_group.draw(self.screen)

        # Display texts
        i = 0
        for enemy in self.enemy_group.sprites():
            if isinstance(enemy, Spaceship):
                enemy_health_text = "Enemy " + str(i) + " Health: " + str(enemy.health)
                red_health_text = self.health_font.render(enemy_health_text, True, WHITE_COLOR)
                self.screen.blit(red_health_text, (WIDTH - red_health_text.get_width() - 10, 15 * (i + 1)))
            i += 1

        yellow_health_text = self.health_font.render(
            "Player Health: " + str(self.player.health), True, WHITE_COLOR)

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
        print(game.AdditionalState())
        if len(game.AdditionalState().shape) < 1:
            print("Warn: " + str(game.AdditionalState().shape))

    game.Exit()
