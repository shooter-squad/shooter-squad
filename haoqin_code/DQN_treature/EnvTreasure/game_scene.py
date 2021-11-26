import math
import random
from typing import Tuple

import pygame
import numpy as np

from .constants import *
from .health_pack import HealthPack
from .obstacle import Obstacle
from .spaceship import Spaceship, SpaceshipType
from .ultimate_ability import UltimateAbility


# os.environ["SDL_VIDEODRIVER"] = "dummy"


def calculate_distance(first: pygame.sprite.Sprite, second: pygame.sprite.Sprite) -> float:
    """
    Calculates the distance between two sprite using their center positions.
    """
    dist = math.sqrt((first.rect.centerx - second.rect.centerx) ** 2 + (first.rect.centery - second.rect.centery) ** 2)
    return dist


def generate_state_vector(spaceship: Spaceship, vec_len: int) -> np.ndarray:
    """
    Generates a state vector for one spaceship given vector length.
    """
    res = [-1 for _ in range(vec_len)]
    if spaceship.is_dead():
        return np.array(res)

    res[0] = spaceship.rect.centerx
    res[1] = spaceship.rect.centery
    i = 2
    for bullet in spaceship.bullets.sprites():
        res[i] = bullet.rect.centerx
        res[i + 1] = bullet.rect.centery
        i += 2
    return np.array(res)

class GameScene(object):
    """
    Our shooter game wrapped in a class.
    YELLOW is the AI player. RED is the enemy.
    Methods that start with upper case will be used by Env.
    """

    def __init__(self):
        pygame.font.init()

        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption(TITLE)

        self.player_border_rect = pygame.Rect(100, OBSTACLE_Y_MAX, WIDTH - 200, HEIGHT - OBSTACLE_Y_MAX)

        self.health_font = pygame.font.SysFont(HEALTH_FONT[0], HEALTH_FONT[1])
        self.winner_font = pygame.font.SysFont(WINNER_FONT[0], WINNER_FONT[1])

        # Load images
        self.yellow_spaceship_image = pygame.transform.scale(pygame.image.load(YELLOW_SPACESHIP_IMAGE_PATH),
                                                             (SPACESHIP_WIDTH, SPACESHIP_HEIGHT))
        self.red_spaceship_image = pygame.transform.rotate(
            pygame.transform.scale(pygame.image.load(RED_SPACESHIP_IMAGE_PATH), (SPACESHIP_WIDTH, SPACESHIP_HEIGHT)),
            180)
        self.blue_spaceship_image = pygame.transform.rotate(
            pygame.transform.scale(pygame.image.load(BLUE_SPACESHIP_IMAGE_PATH), (SPACESHIP_WIDTH, SPACESHIP_HEIGHT)),
            180)

        self.yellow_shielded_image = pygame.transform.scale(pygame.image.load(YELLOW_SPACESHIP_SHIELDED_IMAGE_PATH),
                                                            (SHIELD_WIDTH, SHIELD_HEIGHT))
        self.red_shielded_image = pygame.transform.rotate(
            pygame.transform.scale(pygame.image.load(RED_SPACESHIP_SHIELDED_IMAGE_PATH), (SHIELD_WIDTH, SHIELD_HEIGHT)),
            180)

        self.obstacle_image = pygame.transform.scale(pygame.image.load(OBSTACLE_IMAGE_PATH),
                                                     (OBSTACLE_WIDTH, OBSTACLE_HEIGHT))
        self.health_pack_image = pygame.transform.scale(pygame.image.load(HEALTH_PACK_IMAGE_PATH),
                                                        (HEALTH_PACK_WIDTH, HEALTH_PACK_HEIGHT))

        self.yellow_ultimate_ability_image = pygame.transform.scale(
            pygame.image.load(YELLOW_ULTIMATE_ABILITY_IMAGE_PATH),
            (ULTIMATE_ABILITY_WIDTH, ULTIMATE_ABILITY_HEIGHT))
        self.red_ultimate_ability_image = pygame.transform.scale(pygame.image.load(RED_ULTIMATE_ABILITY_IMAGE_PATH),
                                                                 (ULTIMATE_ABILITY_WIDTH, ULTIMATE_ABILITY_HEIGHT))
        self.blue_ultimate_ability_image = pygame.transform.scale(pygame.image.load(BLUE_ULTIMATE_ABILITY_IMAGE_PATH),
                                                                  (ULTIMATE_ABILITY_WIDTH, ULTIMATE_ABILITY_HEIGHT))

        # =========================== Reset ===========================
        if PURE_COLOR_DISPLAY:
            self.background = pygame.Surface((WIDTH, HEIGHT)).convert()
        else:
            self.background = pygame.Surface((WIDTH, HEIGHT)).convert()

        self.enemy_bullet_group = pygame.sprite.Group()
        self.player_bullet_group = pygame.sprite.Group()

        self.player = Spaceship(
            image=self.yellow_spaceship_image,
            screen_rect=self.screen.get_rect(),
            border_rect=self.player_border_rect,
            bullet_group=self.player_bullet_group,
            shielded_image=self.yellow_shielded_image,
            ultimate_ability_image=self.yellow_ultimate_ability_image,
            start_health=PLAYER_START_HEALTH,
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
                border_rect=self.screen.get_rect(),
                bullet_group=self.enemy_bullet_group,
                start_health=ENEMY_START_HEALTH,
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
        self.total_reward = 0
        self.frame_count = 0

        # * adding action_num
        self.player_action_num = 0
        self.player_bullet_max = 0
        self.enemy_bullet_max = 0

        
        # =========================== Reset ===========================

        # self.Reset()

    # ------------------------- Env wrapper methods -------------------------

    def ActionCount(self):
        return len(Action)

    def ScreenShot(self) -> np.ndarray:
        pixels_arr = pygame.surfarray.array3d(self.screen)
        return pixels_arr

    def StateVector(self, extra_padding=False) -> np.ndarray:
        # print('Width ', WIDTH - 200)
        # print('Height ', HEIGHT - OBSTACLE_Y_MAX)

        # print('Y MAX ', OBSTACLE_Y_MAX)
        # print('Y HEIGHT', HEIGHT - OBSTACLE_Y_MAX)
        # print('player x ', self.player.rect.centerx)
        # print('player y ', self.player.rect.centery)
        # if len(self.player_bullet_group) > self.player_bullet_max:
        #     self.player_bullet_max = len(self.player_bullet_group)
        # if len(self.enemy_bullet_group) > self.enemy_bullet_max:
        #     self.enemy_bullet_max = len(self.enemy_bullet_group)
        # print('Max bullet count')
        # print(self.player_bullet_max)
        # print(self.enemy_bullet_max)

        # * generate player spaceship vector
        player_pos_x_norm = (self.player.rect.centerx - 100 - SPACESHIP_WIDTH / 2 ) / (WIDTH - 200 - SPACESHIP_WIDTH)
        player_pos_y_norm = (self.player.rect.centery - OBSTACLE_Y_MAX - SPACESHIP_HEIGHT / 2) / (HEIGHT - OBSTACLE_Y_MAX - SPACESHIP_HEIGHT)
        space_vector = [player_pos_x_norm, player_pos_y_norm]

        # * generate enemy spaceship vector
        for enemy in self.enemy_group.sprites():
            if isinstance(enemy, Spaceship) and enemy.type == SpaceshipType.NORMAL_ENEMY:
                enemy_pos_x_norm = (enemy.rect.centerx - SPACESHIP_WIDTH / 2) / (WIDTH - SPACESHIP_WIDTH)
                enemy_pos_y_norm = (enemy.rect.centery - SPACESHIP_HEIGHT / 2) / (220)
                space_vector += [enemy_pos_x_norm, enemy_pos_y_norm]

        # * generate player bullet vector
        # player_bullet_vector = [-1 for _ in range(STATE_VECTOR_MAX_BULLET_PER_SPACESHIP * 2)]
        player_bullet_vector = [-1 for _ in range(4 * 2)]
        i = 0
        for bullet in self.player_bullet_group.sprites():
            bullet_pos_x_norm = bullet.rect.centerx / WIDTH
            bullet_pos_y_norm = bullet.rect.centery / HEIGHT
            player_bullet_vector[i] = bullet_pos_x_norm
            player_bullet_vector[i + 1] = bullet_pos_y_norm
            i += 2 

        # * generate enemy bullet vector
        # enemy_bullet_vector = [-1 for _ in range(STATE_VECTOR_MAX_BULLET_PER_SPACESHIP * NORMAL_ENEMY_COUNT * 2)]
        enemy_bullet_vector = [-1 for _ in range(5 * 2)]
        i = 0
        for bullet in self.enemy_bullet_group.sprites():
            bullet_pos_x_norm = bullet.rect.centerx / WIDTH
            bullet_pos_y_norm = bullet.rect.centery / HEIGHT
            enemy_bullet_vector[i] = bullet_pos_x_norm
            enemy_bullet_vector[i + 1] = bullet_pos_y_norm
            i += 2 

        # * generate health_pack vector
        i = 0
        health_pack_vector = [-1 for _ in range(HEALTH_PACK_MAX_COUNT * 2)]
        for health_pack in self.health_pack_group.sprites():
            health_pack_pos_x_norm = (health_pack.rect.centerx - 100 - HEALTH_PACK_WIDTH / 2 ) / (WIDTH - 200 - HEALTH_PACK_WIDTH)
            health_pack_pos_y_norm = (health_pack.rect.centery - OBSTACLE_Y_MAX - HEALTH_PACK_HEIGHT / 2) / (HEIGHT - OBSTACLE_Y_MAX - HEALTH_PACK_HEIGHT)
            health_pack_vector[i] = health_pack_pos_x_norm
            health_pack_vector[i + 1] = health_pack_pos_y_norm
            i += 2

        # print('DEBUG......')
        # print(space_vector)
        # print(player_bullet_vector)
        # print(enemy_bullet_vector)
        # print(health_pack_vector)

        state_arr = np.array(space_vector + player_bullet_vector + enemy_bullet_vector + health_pack_vector)
        return state_arr


    def Done(self):
        return self.done

    def Reward(self):
        return self.reward

    def Reset(self):
        if PURE_COLOR_DISPLAY:
            self.background = pygame.Surface((WIDTH, HEIGHT)).convert()
        else:
            self.background = pygame.Surface((WIDTH, HEIGHT)).convert()

        self.enemy_bullet_group = pygame.sprite.Group()
        self.player_bullet_group = pygame.sprite.Group()

        self.player = Spaceship(
            image=self.yellow_spaceship_image,
            screen_rect=self.screen.get_rect(),
            border_rect=self.player_border_rect,
            bullet_group=self.player_bullet_group,
            shielded_image=self.yellow_shielded_image,
            ultimate_ability_image=self.yellow_ultimate_ability_image,
            start_health=PLAYER_START_HEALTH,
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
                border_rect=self.screen.get_rect(),
                bullet_group=self.enemy_bullet_group,
                start_health=ENEMY_START_HEALTH,
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
        self.total_reward = 0
        self.frame_count = 0

        # * adding action_num
        self.player_action_num = 0

    def Play(self, player_action_num: int):
        # End game if game too long
        if self.frame_count > 5000:
            self.done = True
            return True

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
            if keys_pressed[pygame.K_SPACE]:
                player_action_num = 3
            if keys_pressed[pygame.K_UP]:
                player_action_num = 4
            if keys_pressed[pygame.K_DOWN]:
                player_action_num = 5
            # if keys_pressed[pygame.K_q]:
            #     player_action_num = 6
            # if keys_pressed[pygame.K_w]:
            #     player_action_num = 7

        if player_action_num == -1:
            player_action_num = 0

        # * add player_action_num
        self.player_action_num = player_action_num

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
        self.total_reward += self.reward
        self.draw_window()
        self.clock.tick(FPS)

        return False

    def AdditionalState(self) -> np.ndarray:
        """
        Returns additional state parameters
        """
        player_arr = np.zeros(ADDITIONAL_STATE_LEN_PLAYER, dtype=np.float32)
        normal_arr = np.zeros(ADDITIONAL_STATE_LEN_NORMAL, dtype=np.float32)
        charge_arr = np.zeros(ADDITIONAL_STATE_LEN_CHARGE, dtype=np.float32)

        player_temp = np.array([
            self.player.health / PLAYER_START_HEALTH,
            self.player.get_shield_cool_down() / SHIELD_COOL_DOWN,
            int(self.player.ultimate_available)
        ])
        player_arr[0:player_temp.shape[0]] = player_temp

        normal_temp = []
        charge_temp = []
        for enemy in self.enemy_group.sprites():
            if isinstance(enemy, Spaceship) and enemy.type == SpaceshipType.NORMAL_ENEMY:
                normal_temp += [
                    enemy.health / ENEMY_START_HEALTH,
                    enemy.get_shield_cool_down() / SHIELD_COOL_DOWN,
                    int(enemy.ultimate_available)
                ]
            elif isinstance(enemy, Spaceship) and enemy.type == SpaceshipType.CHARGE_ENEMY:
                charge_temp += [
                    enemy.health / ENEMY_START_HEALTH,
                    enemy.get_shield_cool_down() / SHIELD_COOL_DOWN,
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
        if len(self.health_pack_group.sprites()) >= HEALTH_PACK_MAX_COUNT:
            return

        
        pos_x = None
        pos_y = None
        
        for j in range(10):
            separate = True
            pos_x=random.randrange(self.player_border_rect.left, self.player_border_rect.right - HEALTH_PACK_WIDTH,
                               HEALTH_PACK_WIDTH // 3)
            if (abs(pos_x - self.player.rect.x) < 200):
                separate = False
            for health_pack in self.health_pack_group.sprites():
                if (abs(pos_x - health_pack.rect.x) < 60):
                    separate = False
            if separate:
                break

        for j in range(10):
            separate = True
            pos_y=random.randrange(self.player_border_rect.top, self.player_border_rect.bottom - HEALTH_PACK_HEIGHT,
                                HEALTH_PACK_HEIGHT // 3)
            # if (abs(pos_y - self.player.rect.y) < 60):
            #     separate = False
            for health_pack in self.health_pack_group.sprites():
                if (abs(pos_y - health_pack.rect.y) < 60):
                    separate = False
            if separate:
                break

        health_pack = HealthPack(
            image=self.health_pack_image,
            x=pos_x,
            y=pos_y
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
            border_rect=self.screen.get_rect(),
            bullet_group=self.enemy_bullet_group,
            start_health=ENEMY_START_HEALTH,
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
        # if enemy.ultimate_available and calculate_distance(enemy, self.player) <= ULTIMATE_ABILITY_WIDTH / 2:
        #     return Action.USE_ULTIMATE_ABILITY

        if calculate_distance(enemy, self.player) <= SPACESHIP_WIDTH * 2:
            if enemy.enemy_behavior == Action.RIGHT and self.player.rect.centerx < enemy.rect.centerx:
                enemy.enemy_behavior = Action.LEFT
            elif enemy.enemy_behavior == Action.LEFT and self.player.rect.centerx > enemy.rect.centerx:
                enemy.enemy_behavior = Action.RIGHT

        if enemy.enemy_behavior == Action.RIGHT and enemy.rect.left <= 0:
            enemy.enemy_behavior = Action.LEFT
        if enemy.enemy_behavior == Action.LEFT and enemy.rect.right >= WIDTH:
            enemy.enemy_behavior = Action.RIGHT
        # fire_or_shield = Action.ACTIVATE_SHIELD if enemy.shield_enabled and enemy.get_shield_cool_down() == 0 else Action.FIRE
        fire_or_shield = Action.FIRE
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
        # print("Each frame")
        player_action = Action(player_action_num)
        self.player.update(player_action, self.obstacle_group.sprites() + self.enemy_group.sprites())
        self.player_bullet_group.update()
        self.player.ultimate_abilities.update()

        # Enemy action is calculated
        i = 0
        self.enemy_bullet_group.update()
        for enemy in self.enemy_group.sprites():
            if isinstance(enemy, Spaceship):
                if not enemy.is_dead():
                    enemy_action = self.calculate_enemy_action(enemy, i)
                    enemy.update(enemy_action, self.obstacle_group.sprites() + [self.player])
                enemy.ultimate_abilities.update()
            i += 1

        

        # Spawn health pack if time is reached
        if self.frame_count == CHARGE_ENEMY_SPAWN_INTERVAL and CHARGE_ENEMY_ENABLED:
            self.spawn_charge_enemy()

        # Check collisions:

        # 1) Player vs enemy bullets
        hit_list = pygame.sprite.spritecollide(self.player, self.enemy_bullet_group, True)
        if not self.player.shield_activated:
            self.player.health -= BULLET_DAMAGE * len(hit_list)
            self.reward += Reward.BULLET_HIT_PLAYER.value * len(hit_list)

        for enemy in self.enemy_group.sprites():
            if isinstance(enemy, Spaceship):
                # 2) Enemy vs player bullets
                if not enemy.is_dead():
                    hit_list = pygame.sprite.spritecollide(enemy, self.player_bullet_group, True)
                    if not enemy.shield_activated:
                        enemy.health -= BULLET_DAMAGE * len(hit_list)
                        self.reward += Reward.BULLET_HIT_ENEMY.value * len(hit_list)

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
        pygame.sprite.groupcollide(self.obstacle_group, self.player_bullet_group, False, True)
        pygame.sprite.groupcollide(self.obstacle_group, self.enemy_bullet_group, False, True)

        # 4) Player vs health pack
        hit_list = pygame.sprite.spritecollide(self.player, self.health_pack_group, True)
        # self.player.health += HEALTH_PACK_HEALTH_RECOVERED * len(hit_list)
        self.reward += Reward.PLAYER_GET_HEALTH_PACK.value * len(hit_list)

         # Spawn health pack if time is reached
        if HEALTH_PACK_ENABLED and len(
                self.health_pack_group.sprites()) < HEALTH_PACK_MAX_COUNT:  # and self.frame_count % HEALTH_PACK_TIME_INTERVAL == 0:
            self.spawn_health_pack()

        if True:  # self.frame_count % NORMAL_ENEMY_SPAWN_INTERVAL == 0:
            for enemy in self.enemy_group.sprites():
                if isinstance(enemy, Spaceship):
                    if enemy.is_dead():
                        start_x = random.randrange(0, WIDTH - SPACESHIP_WIDTH)
                        enemy.start_x = start_x
                        enemy.reset()
                        enemy.enemy_behavior = Action.LEFT if random.random() < 0.5 else Action.RIGHT

        if NEGATIVE_REWARD_ENABLED:
            self.reward -= NEGATIVE_REWARD

        # print(self.reward)

    def draw_window(self):
        self.screen.blit(self.background, (0, 0))

        # Draw player
        self.player.ultimate_abilities.draw(self.screen)
        self.player_group.draw(self.screen)
        self.player_bullet_group.draw(self.screen)

        # Draw enemy
        self.enemy_bullet_group.draw(self.screen)
        for enemy in self.enemy_group.sprites():
            if isinstance(enemy, Spaceship):
                enemy.ultimate_abilities.draw(self.screen)
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

        reward_text = self.health_font.render(
            "Reward: " + str(self.total_reward), True, WHITE_COLOR)
        self.screen.blit(reward_text, (WIDTH // 2 - 15, 10))

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
        # print(game.AdditionalState())
        sv = game.StateVector()
        if len(game.AdditionalState().shape) < 1:
            print("Warn: " + str(game.AdditionalState().shape))

    game.Exit()
