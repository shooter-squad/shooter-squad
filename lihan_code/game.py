import pygame
import os
from enum import Enum
from gym.spaces import Discrete


class Action(Enum):
    """
    All actions in the game.
    """
    NOOP = 0
    LEFT = 1
    RIGHT = 2
    FIRE = 3


class Game(object):
    """
    Our shooter game wrapped in a class.
    YELLOW is the AI player. RED is the enemy.
    """

    def __init__(self):
        pygame.font.init()
        pygame.mixer.init()

        self.WIDTH, self.HEIGHT = 900, 500
        self.WIN = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("First Game!")

        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.YELLOW = (255, 255, 0)

        self.BORDER = pygame.Rect(self.WIDTH // 2 - 5, 0, 10, self.HEIGHT)

        # self.BULLET_HIT_SOUND = pygame.mixer.Sound('Assets/Grenade+1.mp3')
        # self.BULLET_FIRE_SOUND = pygame.mixer.Sound('Assets/Gun+Silencer.mp3')

        self.HEALTH_FONT = pygame.font.SysFont('comicsans', 40)
        self.WINNER_FONT = pygame.font.SysFont('comicsans', 100)

        self.FPS = 20  # Using a lower FPS because of the model
        self.VEL = 5
        self.BULLET_VEL = 7
        self.MAX_BULLETS = 3
        self.SPACESHIP_WIDTH, self.SPACESHIP_HEIGHT = 55, 40

        self.YELLOW_HIT = pygame.USEREVENT + 1
        self.RED_HIT = pygame.USEREVENT + 2

        self.YELLOW_SPACESHIP_IMAGE = pygame.image.load(
            os.path.join('../Assets', 'spaceship_yellow.png'))
        self.YELLOW_SPACESHIP = pygame.transform.scale(self.YELLOW_SPACESHIP_IMAGE,
                                                       (self.SPACESHIP_WIDTH, self.SPACESHIP_HEIGHT))

        self.RED_SPACESHIP_IMAGE = pygame.image.load(
            os.path.join('../Assets', 'spaceship_red.png'))
        self.RED_SPACESHIP = pygame.transform.rotate(pygame.transform.scale(
            self.RED_SPACESHIP_IMAGE, (self.SPACESHIP_WIDTH, self.SPACESHIP_HEIGHT)), 180)

        self.SPACE = pygame.transform.scale(pygame.image.load(
            os.path.join('../Assets', 'space.png')), (self.WIDTH, self.HEIGHT))

        self.red = None
        self.yellow = None

        self.red_bullets = None
        self.yellow_bullets = None

        self.red_health = None
        self.yellow_health = None

        self.clock = None
        self.run = None
        self.reward = None

        self.Reset()

    def Actions(self):
        # NOOP, LEFT, RIGHT, FIRE
        return Discrete(len(Action))

    def ScreenShot(self):
        pixels_arr = pygame.surfarray.array3d(self.WIN)
        return pixels_arr

    def Done(self):
        return not self.run

    def Reward(self):
        return self.reward

    def Reset(self):
        self.red = pygame.Rect(400, 100, self.SPACESHIP_WIDTH, self.SPACESHIP_HEIGHT)
        self.yellow = pygame.Rect(400, 300, self.SPACESHIP_WIDTH, self.SPACESHIP_HEIGHT)

        self.red_bullets = []
        self.yellow_bullets = []

        self.red_health = 10
        self.yellow_health = 10

        self.clock = pygame.time.Clock()
        self.run = True

    def Play(self, action_num):
        if self.Done():
            return

        # Handle player health and reward
        current_reward = 0
        for event in pygame.event.get():

            # Human inputs
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()

            # if event.type == pygame.KEYDOWN:
            #     if event.key == pygame.K_LCTRL and len(self.yellow_bullets) < self.MAX_BULLETS:
            #         bullet = pygame.Rect(
            #             self.yellow.x + self.yellow.width // 2 - 2, self.yellow.y, 5, 10)
            #         self.yellow_bullets.append(bullet)
            #         # self.BULLET_FIRE_SOUND.play()
            #
            #     if event.key == pygame.K_RCTRL and len(self.red_bullets) < self.MAX_BULLETS:
            #         bullet = pygame.Rect(
            #             self.red.x + self.red.width // 2 - 2, self.red.y + self.red.height, 5, 10)
            #         self.red_bullets.append(bullet)
            #         # self.BULLET_FIRE_SOUND.play()

            if event.type == self.RED_HIT:
                self.red_health -= 1
                current_reward += 10
                # self.BULLET_HIT_SOUND.play()

            if event.type == self.YELLOW_HIT:
                self.yellow_health -= 1
                current_reward -= 10
                # self.BULLET_HIT_SOUND.play()

        self.reward = current_reward

        # Check if game is over
        winner_text = ""
        if self.red_health <= 0:
            winner_text = "Yellow Wins!"
            self.run = False

        if self.yellow_health <= 0:
            winner_text = "Red Wins!"
            # self.run = False

        if winner_text != "":
            self.draw_winner(winner_text)
            return

        # Do action
        keys_pressed = {
            pygame.K_a: False,
            pygame.K_d: False,
            pygame.K_w: False,
            pygame.K_s: False,
            pygame.K_UP: False,
            pygame.K_DOWN: False,
            pygame.K_LEFT: False,
            pygame.K_RIGHT: False,
        }

        action = Action(action_num)
        if action == Action.FIRE:
            bullet = pygame.Rect(
                self.yellow.x + self.yellow.width // 2 - 2, self.yellow.y, 5, 10)
            self.yellow_bullets.append(bullet)
        elif action == Action.LEFT:
            keys_pressed[pygame.K_a] = True
        elif action == Action.RIGHT:
            keys_pressed[pygame.K_d] = True

        # Update display
        self.yellow_handle_movement(keys_pressed, self.yellow)
        self.red_handle_movement(keys_pressed, self.red)

        self.handle_bullets(self.yellow_bullets, self.red_bullets, self.yellow, self.red)

        self.draw_window(self.red, self.yellow, self.red_bullets, self.yellow_bullets,
                         self.red_health, self.yellow_health)

        self.clock.tick(self.FPS)

    # ------------------------- Display update methods -------------------------

    def draw_window(self, red, yellow, red_bullets, yellow_bullets, red_health, yellow_health):
        self.WIN.blit(self.SPACE, (0, 0))
        # pygame.draw.rect(self.WIN, self.BLACK, self.BORDER)

        red_health_text = self.HEALTH_FONT.render(
            "Health: " + str(red_health), 1, self.WHITE)
        yellow_health_text = self.HEALTH_FONT.render(
            "Health: " + str(yellow_health), 1, self.WHITE)
        self.WIN.blit(red_health_text, (self.WIDTH - red_health_text.get_width() - 10, 10))
        self.WIN.blit(yellow_health_text, (10, 10))

        self.WIN.blit(self.YELLOW_SPACESHIP, (yellow.x, yellow.y))
        self.WIN.blit(self.RED_SPACESHIP, (red.x, red.y))

        for bullet in red_bullets:
            pygame.draw.rect(self.WIN, self.RED, bullet)

        for bullet in yellow_bullets:
            pygame.draw.rect(self.WIN, self.YELLOW, bullet)

        pygame.display.update()

    def yellow_handle_movement(self, keys_pressed, yellow):
        if keys_pressed[pygame.K_a] and yellow.x - self.VEL > 0:  # LEFT
            yellow.x -= self.VEL
        if keys_pressed[pygame.K_d]:  # RIGHT
            yellow.x += self.VEL
        if keys_pressed[pygame.K_w] and yellow.y - self.VEL > 0:  # UP
            yellow.y -= self.VEL
        if keys_pressed[pygame.K_s] and yellow.y + self.VEL + yellow.height < self.HEIGHT - 15:  # DOWN
            yellow.y += self.VEL

    def red_handle_movement(self, keys_pressed, red):
        if keys_pressed[pygame.K_LEFT]:  # LEFT
            red.x -= self.VEL
        if keys_pressed[pygame.K_RIGHT] and red.x + self.VEL + red.width < self.WIDTH:  # RIGHT
            red.x += self.VEL
        if keys_pressed[pygame.K_UP] and red.y - self.VEL > 0:  # UP
            red.y -= self.VEL
        if keys_pressed[pygame.K_DOWN] and red.y + self.VEL + red.height < self.HEIGHT - 15:  # DOWN
            red.y += self.VEL

    def handle_bullets(self, yellow_bullets, red_bullets, yellow, red):
        for bullet in yellow_bullets:
            bullet.y -= self.BULLET_VEL
            if red.colliderect(bullet):
                pygame.event.post(pygame.event.Event(self.RED_HIT))
                yellow_bullets.remove(bullet)
            elif bullet.x > self.WIDTH or bullet.y > self.HEIGHT or bullet.x < 0 or bullet.y < 0:
                yellow_bullets.remove(bullet)

        for bullet in red_bullets:
            bullet.y += self.BULLET_VEL
            if yellow.colliderect(bullet):
                pygame.event.post(pygame.event.Event(self.YELLOW_HIT))
                red_bullets.remove(bullet)
            elif bullet.x > self.WIDTH or bullet.y > self.HEIGHT or bullet.x < 0 or bullet.y < 0:
                red_bullets.remove(bullet)

    def draw_winner(self, text):
        draw_text = self.WINNER_FONT.render(text, 1, self.WHITE)
        self.WIN.blit(draw_text, (self.WIDTH / 2 - draw_text.get_width() /
                                  2, self.HEIGHT / 2 - draw_text.get_height() / 2))
        pygame.display.update()
        pygame.time.delay(5000)


if __name__ == "__main__":
    game = Game()
    action_list = [
        0, 1, 2, 3, 0, 0, 1, 2, 3, 0, 0, 1, 2, 3, 0, 0, 1, 2, 3, 0, 0, 1, 2, 3, 0
    ]

    for action_num in action_list:
        if game.Done():
            break
        game.Play(action_num)
        game.Play(action_num)
        game.Play(action_num)
        game.Play(action_num)
        game.Play(action_num)
