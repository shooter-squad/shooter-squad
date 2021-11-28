import os
from enum import Enum

package_directory = os.path.dirname(os.path.abspath(__file__))


# ------------------------- Parameters -------------------------


class Action(Enum):
    """
    All actions in the game.
    """
    NOOP = 0
    LEFT = 1
    RIGHT = 2
    FIRE = 3
    UP = 4
    DOWN = 5
    # ACTIVATE_SHIELD = 6
    # USE_ULTIMATE_ABILITY = 7


class Reward(Enum):
    """
    All rewards in the game
    """
    BULLET_HIT_ENEMY = 15
    BULLET_HIT_PLAYER = -35
    PLAYER_GET_HEALTH_PACK = 10
    ULTIMATE_HIT_ENEMY = 10
    ULTIMATE_HIT_PLAYER = -10
    PLAYER_HIT_CHARGE_ENEMY = -100


PURE_COLOR_DISPLAY = False
NEGATIVE_REWARD_ENABLED = False
NEGATIVE_REWARD = 0.005
MAX_FRAME_PER_GAME = 5000

# ------------------------- Values -------------------------
TITLE = "AI bot"

WIDTH = 800
HEIGHT = 800
FPS = 30
VEL = 5

ENEMY_COUNT = 3
NORMAL_ENEMY_COUNT = 2
CHARGE_ENEMY_COUNT = 1
WIN_COUNT = 2
ENEMY_START_Y_RANGES = [[0, 100], [130, 220]]
CHARGE_ENEMY_SPAWN_INTERVAL = 600
CHARGE_ENEMY_ENABLED = False

ADDITIONAL_STATE_LEN_PLAYER = 6
ADDITIONAL_STATE_LEN_NORMAL = 18
ADDITIONAL_STATE_LEN_CHARGE = 3

STATE_VECTOR_MAX_BULLET_PER_SPACESHIP = 5

BULLET_WIDTH = 16
BULLET_HEIGHT = 24
BULLET_VEL = 10
MAX_BULLETS = 3
BULLET_DAMAGE = 1
ENEMY_BULLET_INTERVAL = 40
PLAYER_BULLET_INTERVAL = 20

SPACESHIP_WIDTH = 60
SPACESHIP_HEIGHT = 72
SHIELD_WIDTH = 118
SHIELD_HEIGHT = 114
ENEMY_START_HEALTH = 2
PLAYER_START_HEALTH = 10
NORMAL_ENEMY_SPAWN_INTERVAL = 50

RED_START_POSITION = (375, 80)
YELLOW_START_POSITION = (375, 650)

SHIELD_DURATION = 60
SHIELD_COOL_DOWN = 120  # Time since last time shield was activated
ENEMY_SHIELD_ENABLED = False

OBSTACLE_COUNT = 2
OBSTACLE_WIDTH = 100
OBSTACLE_HEIGHT = 60
OBSTACLE_Y_MIN = 300
OBSTACLE_Y_MAX = 550

HEALTH_PACK_ENABLED = True
HEALTH_PACK_WIDTH = 40
HEALTH_PACK_HEIGHT = 40
HEALTH_PACK_HEALTH_RECOVERED = 1
HEALTH_PACK_TIME_INTERVAL = 50
HEALTH_PACK_MAX_COUNT = 5

ULTIMATE_ABILITY_WIDTH = 400
ULTIMATE_ABILITY_HEIGHT = 400
ULTIMATE_ABILITY_MAX_DAMAGE = 2
ULTIMATE_ABILITY_DAMAGE_INTERVAL = 10

YELLOW_SPACESHIP_IMAGE_PATH = os.path.join(package_directory, 'Assets', 'spaceship_yellow.png')
RED_SPACESHIP_IMAGE_PATH = os.path.join(package_directory, 'Assets', 'spaceship_red.png')
BLUE_SPACESHIP_IMAGE_PATH = os.path.join(package_directory, 'Assets', 'spaceship_blue.png')
GREEN_SPACESHIP_IMAGE_PATH = os.path.join(package_directory, 'Assets', 'spaceship_green.png')
YELLOW_SPACESHIP_SHIELDED_IMAGE_PATH = os.path.join(package_directory, 'Assets', 'spaceship_yellow_shielded.png')
RED_SPACESHIP_SHIELDED_IMAGE_PATH = os.path.join(package_directory, 'Assets', 'spaceship_red_shielded.png')
HEALTH_PACK_IMAGE_PATH = os.path.join(package_directory, 'Assets', 'treasure.png')
OBSTACLE_IMAGE_PATH = os.path.join(package_directory, 'Assets', 'obstacle.png')
YELLOW_ULTIMATE_ABILITY_IMAGE_PATH = os.path.join(package_directory, 'Assets', 'yellow_ultimate_ability.png')
RED_ULTIMATE_ABILITY_IMAGE_PATH = os.path.join(package_directory, 'Assets', 'red_ultimate_ability.png')
BLUE_ULTIMATE_ABILITY_IMAGE_PATH = os.path.join(package_directory, 'Assets', 'blue_ultimate_ability.png')
GREEN_ULTIMATE_ABILITY_IMAGE_PATH = os.path.join(package_directory, 'Assets', 'green_ultimate_ability.png')
SPACE_IMAGE_PATH = os.path.join(package_directory, 'Assets', 'space.png')

# ------------------------- Colors -------------------------
WHITE_COLOR = (255, 255, 255)
RED_COLOR = (255, 0, 0)
YELLOW_COLOR = (255, 255, 0)
BLUE_COLOR = (0, 162, 232)

# ------------------------- Fonts -------------------------
HEALTH_FONT = ('comicsans', 20)
WINNER_FONT = ('comicsans', 100)
