import os
from enum import Enum


# ------------------------- Parameters -------------------------
class Action(Enum):
    """
    All actions in the game.
    """
    NOOP = 0
    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4
    FIRE = 5
    ACTIVATE_SHIELD = 6


class Reward(Enum):
    """
    All rewards in the game
    """
    BULLET_HIT_ENEMY = 10
    BULLET_HIT_PLAYER = -10
    PLAYER_GET_HEALTH_PACK = 10


PURE_COLOR_DISPLAY = False
NEGATIVE_REWARD_ENABLED = True
NEGATIVE_REWARD = 0.005

# ------------------------- Values -------------------------
TITLE = "AI bot"

WIDTH = 800
HEIGHT = 800
FPS = 60
VEL = 5
ENEMY_COUNT = 1

BULLET_WIDTH = 10
BULLET_HEIGHT = 15
BULLET_VEL = 10
MAX_BULLETS = 3
BULLET_DAMAGE = 1
BULLET_INTERVAL = 20

SPACESHIP_WIDTH = 60
SPACESHIP_HEIGHT = 72
SHIELD_WIDTH = 118
SHIELD_HEIGHT = 114
RED_START_HEALTH = 10
YELLOW_START_HEALTH = 10

RED_START_POSITION = (375, 80)
YELLOW_START_POSITION = (375, 650)

SHIELD_DURATION = 60
SHIELD_COOL_DOWN = 120  # Time since last time shield was activated

OBSTACLE_COUNT = 3
OBSTACLE_WIDTH = 100
OBSTACLE_HEIGHT = 60
OBSTACLE_Y_MIN = 300
OBSTACLE_Y_MAX = 550

HEALTH_PACK_WIDTH = 40
HEALTH_PACK_HEIGHT = 40
HEALTH_PACK_HEALTH_RECOVERED = 1
HEALTH_PACK_TIME_INTERVAL = 400

YELLOW_SPACESHIP_IMAGE_PATH = os.path.join('../../Assets', 'spaceship_yellow.png')
RED_SPACESHIP_IMAGE_PATH = os.path.join('../../Assets', 'spaceship_red.png')
YELLOW_SPACESHIP_SHIELDED_IMAGE_PATH = os.path.join('../../Assets', 'spaceship_yellow_shielded.png')
RED_SPACESHIP_SHIELDED_IMAGE_PATH = os.path.join('../../Assets', 'spaceship_red_shielded.png')
HEALTH_PACK_IMAGE_PATH = os.path.join('../../Assets', 'health_pack.png')
OBSTACLE_IMAGE_PATH = os.path.join('../../Assets', 'obstacle.png')
SPACE_IMAGE_PATH = os.path.join('../../Assets', 'space.png')
HIT_SOUND_PATH = os.path.join('../../Assets', "Grenade+1.mp3")
FIRE_SOUND_PATH = os.path.join('../../Assets', "Gun+Silencer.mp3")

# ------------------------- Colors -------------------------
WHITE_COLOR = (255, 255, 255)
RED_COLOR = (255, 0, 0)
YELLOW_COLOR = (255, 255, 0)

# ------------------------- Fonts -------------------------
HEALTH_FONT = ('comicsans', 40)
WINNER_FONT = ('comicsans', 100)
