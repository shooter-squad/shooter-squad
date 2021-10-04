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
    TURN_COUNTERCLOCKWISE = 6
    TURN_CLOCKWISE = 7


class Reward(Enum):
    """
    All rewards in the game
    """
    BULLET_HIT_ENEMY = 10
    BULLET_HIT_PLAYER = -10


# ------------------------- Values -------------------------
TITLE = "AI bot"

WIDTH = 900
HEIGHT = 500
FPS = 60
VEL = 5
THETA = 30

BULLET_WIDTH = 5
BULLET_HEIGHT = 5
BULLET_VEL = 7
MAX_BULLETS = 3
BULLET_DAMAGE = 1

SPACESHIP_WIDTH = 55
SPACESHIP_HEIGHT = 40
RED_START_HEALTH = 10
YELLOW_START_HEALTH = 10
FRAME_TO_UPDATE = 10


RED_START_POSITION = (500, 200)
YELLOW_START_POSITION = (300, 200)

YELLOW_SPACESHIP_IMAGE_PATH = os.path.join('../Assets', 'spaceship_yellow.png')
RED_SPACESHIP_IMAGE_PATH = os.path.join('../Assets', 'spaceship_red.png')
SPACE_IMAGE_PATH = os.path.join('../Assets', 'space.png')
HIT_SOUND_PATH = os.path.join('../Assets', "Grenade+1.mp3")
FIRE_SOUND_PATH = os.path.join('../Assets', "Gun+Silencer.mp3")

# ------------------------- Colors -------------------------
WHITE_COLOR = (255, 255, 255)
RED_COLOR = (255, 0, 0)
YELLOW_COLOR = (255, 255, 0)

# ------------------------- Fonts -------------------------
HEALTH_FONT = ('comicsans', 40)
DEBUG_FONT = ('comicsans', 20)
WINNER_FONT = ('comicsans', 100)
