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
    USE_ULTIMATE_ABILITY = 7


class Reward(Enum):
    """
    All rewards in the game
    """
    BULLET_HIT_ENEMY = 10
    BULLET_HIT_PLAYER = -10
    PLAYER_GET_HEALTH_PACK = 10
    ULTIMATE_HIT_ENEMY = 10
    ULTIMATE_HIT_PLAYER = -10
    PLAYER_HIT_CHARGE_ENEMY = -100


PURE_COLOR_DISPLAY = False
NEGATIVE_REWARD_ENABLED = True
NEGATIVE_REWARD = 0.005

# ------------------------- Values -------------------------
TITLE = "AI bot"

WIDTH = 800
HEIGHT = 800
FPS = 60
VEL = 5

ENEMY_COUNT = 3
NORMAL_ENEMY_COUNT = 2
CHARGE_ENEMY_COUNT = 1
WIN_COUNT = 2
ENEMY_START_Y_RANGES = [[0, 100], [130, 220]]
CHARGE_ENEMY_SPAWN_INTERVAL = 600

ADDITIONAL_STATE_LEN_PLAYER = 6
ADDITIONAL_STATE_LEN_NORMAL = 18
ADDITIONAL_STATE_LEN_CHARGE = 3

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
RED_START_HEALTH = 4
YELLOW_START_HEALTH = 10

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

HEALTH_PACK_WIDTH = 40
HEALTH_PACK_HEIGHT = 40
HEALTH_PACK_HEALTH_RECOVERED = 1
HEALTH_PACK_TIME_INTERVAL = 400

ULTIMATE_ABILITY_WIDTH = 400
ULTIMATE_ABILITY_HEIGHT = 400
ULTIMATE_ABILITY_MAX_DAMAGE = 2
ULTIMATE_ABILITY_DAMAGE_INTERVAL = 10

YELLOW_SPACESHIP_IMAGE_PATH = os.path.join('../../Assets', 'spaceship_yellow.png')
RED_SPACESHIP_IMAGE_PATH = os.path.join('../../Assets', 'spaceship_red.png')
BLUE_SPACESHIP_IMAGE_PATH = os.path.join('../../Assets', 'spaceship_blue.png')
GREEN_SPACESHIP_IMAGE_PATH = os.path.join('../../Assets', 'spaceship_green.png')
YELLOW_SPACESHIP_SHIELDED_IMAGE_PATH = os.path.join('../../Assets', 'spaceship_yellow_shielded.png')
RED_SPACESHIP_SHIELDED_IMAGE_PATH = os.path.join('../../Assets', 'spaceship_red_shielded.png')
HEALTH_PACK_IMAGE_PATH = os.path.join('../../Assets', 'health_pack.png')
OBSTACLE_IMAGE_PATH = os.path.join('../../Assets', 'obstacle.png')
YELLOW_ULTIMATE_ABILITY_IMAGE_PATH = os.path.join('../../Assets', 'yellow_ultimate_ability.png')
RED_ULTIMATE_ABILITY_IMAGE_PATH = os.path.join('../../Assets', 'red_ultimate_ability.png')
BLUE_ULTIMATE_ABILITY_IMAGE_PATH = os.path.join('../../Assets', 'blue_ultimate_ability.png')
GREEN_ULTIMATE_ABILITY_IMAGE_PATH = os.path.join('../../Assets', 'green_ultimate_ability.png')
SPACE_IMAGE_PATH = os.path.join('../../Assets', 'space.png')
HIT_SOUND_PATH = os.path.join('../../Assets', "Grenade+1.mp3")
FIRE_SOUND_PATH = os.path.join('../../Assets', "Gun+Silencer.mp3")

# ------------------------- Colors -------------------------
WHITE_COLOR = (255, 255, 255)
RED_COLOR = (255, 0, 0)
YELLOW_COLOR = (255, 255, 0)
BLUE_COLOR = (0, 162, 232)

# ------------------------- Fonts -------------------------
HEALTH_FONT = ('comicsans', 20)
WINNER_FONT = ('comicsans', 100)
