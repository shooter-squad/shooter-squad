import random
import os

import pygame

from .constants import *
from .spaceship import Spaceship

os.environ["SDL_VIDEODRIVER"] = "dummy"

class GameScene(object):
    """
    Our shooter game wrapped in a class.
    YELLOW is the AI player. RED is the enemy.
    Methods that start with upper case will be used by Env.
    """

    def __init__(self):
        pygame.font.init()
        pygame.mixer.init()

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

        if PURE_COLOR_DISPLAY:
            self.background = pygame.Surface((WIDTH, HEIGHT)).convert()
        else:
            self.background = pygame.transform.scale(pygame.image.load(SPACE_IMAGE_PATH), (WIDTH, HEIGHT))

        self.player = Spaceship(
            image=yellow_spaceship_image,
            screen_rect=self.screen.get_rect(),
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
            screen_rect=self.screen.get_rect(),
            start_health=RED_START_HEALTH,
            start_x=RED_START_POSITION[0],
            start_y=RED_START_POSITION[1],
            color=RED_COLOR,
            up_direction=False
        )
        self.enemy_group = pygame.sprite.Group()
        self.enemy_group.add(self.enemy)

        self.clock = pygame.time.Clock()
        self.run = True
        self.reward = 0
        self.enemy_direction = 'left'
        self.Reset()

    def ActionCount(self):
        return len(Action)

    def ScreenShot(self):
        pixels_arr = pygame.surfarray.array3d(self.screen)
        return pixels_arr

    def Done(self):
        return not self.run

    def Reward(self):
        return self.reward

    def Reset(self):
        self.player.reset()
        self.enemy.reset()
        self.run = True

    def Play(self, player_action_num: int):
        # Human input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if player_action_num == -1:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        player_action_num = 1
                    if event.key == pygame.K_RIGHT:
                        player_action_num = 2
                    if event.key == pygame.K_SPACE:
                        player_action_num = 3

        if not self.run:
            return

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
            self.run = False
            return

        self.reward = 0
        self.update(player_action_num)
        self.draw_window()
        self.clock.tick(FPS)

    def Exit(self):
        pygame.quit()

    # ------------------------- Display update methods -------------------------

    def update(self, player_action_num: int):
        # Player action from input
        player_action = Action(player_action_num)
        self.player.update(player_action, [self.enemy])
        self.player.bullets.update()

        # Enemy action is randomly chosen
        # enemy_action = Action(random.randint(0, len(Action) - 1))
        if self.enemy_direction == 'right':
            if self.enemy.rect.left <= 50:
                self.enemy_direction = 'left'
        if self.enemy_direction == 'left':
            if self.enemy.rect.right >= WIDTH - 50:
                self.enemy_direction = 'right'
        enemy_action = Action.LEFT if self.enemy_direction == 'left' else Action.RIGHT
        enemy_action = enemy_action if random.random() < 0.7 else Action.FIRE

        self.enemy.update(enemy_action, [self.player])
        self.enemy.bullets.update()

        # Check collisions:
        # 1) Player vs enemy bullets
        hit_list = pygame.sprite.spritecollide(self.player, self.enemy.bullets, True)
        self.player.health -= BULLET_DAMAGE * len(hit_list)
        self.reward += Reward.BULLET_HIT_PLAYER.value * len(hit_list)

        # 2) Enemy vs player bullets
        hit_list = pygame.sprite.spritecollide(self.enemy, self.player.bullets, True)
        self.enemy.health -= BULLET_DAMAGE * len(hit_list)
        self.reward += Reward.BULLET_HIT_ENEMY.value * len(hit_list)

        if NEGATIVE_REWARD_ENABLED:
            self.reward -= NEGATIVE_REWARD

    def draw_window(self):
        self.screen.blit(self.background, (0, 0))

        # Display texts
        red_health_text = self.health_font.render(
            "Health: " + str(self.enemy.health), True, WHITE_COLOR)
        yellow_health_text = self.health_font.render(
            "Health: " + str(self.player.health), True, WHITE_COLOR)
        self.screen.blit(red_health_text, (WIDTH - red_health_text.get_width() - 10, 10))
        self.screen.blit(yellow_health_text, (10, 10))

        # Draw player
        self.player_group.draw(self.screen)
        self.player.bullets.draw(self.screen)

        # Draw enemy
        self.enemy_group.draw(self.screen)
        self.enemy.bullets.draw(self.screen)

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
