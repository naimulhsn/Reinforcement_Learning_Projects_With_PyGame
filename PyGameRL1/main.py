import pygame
import random
import numpy as np

FRAME_SIZE = 7  # Actual frame size is Frame_size*2+1
FRAME_UNIT = (FRAME_SIZE * 2 + 1)
PIXEL_UNIT = 32
SCREEN_WIDTH = FRAME_UNIT * PIXEL_UNIT  # 32 * 20 (20 = col)
SCREEN_HEIGHT = FRAME_UNIT * PIXEL_UNIT  # 32 * 15 (15 = row)
BACKGROUND_COLOR = (20, 20, 20)

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
font = pygame.font.Font('freesansbold.ttf', 14)

img_floor = pygame.image.load("floor.png")
img_block = pygame.image.load("block.png")
img_trap = pygame.image.load("trap.png")
img_food = pygame.image.load("food.png")
img_player_left = pygame.image.load("player_left.png")
img_player_right = pygame.image.load("player_right.png")
img_player_up = pygame.image.load("player_up.png")
img_player_down = pygame.image.load("player_down.png")
img_enemy_left = pygame.image.load("enemy_left.png")
img_enemy_right = pygame.image.load("enemy_right.png")
img_enemy_up = pygame.image.load("enemy_up.png")
img_enemy_down = pygame.image.load("enemy_down.png")

GAME_WORLD = np.random.randint(1, 12, (1000, 1000))
GAME_WORLD = np.where(GAME_WORLD > 4, 1, GAME_WORLD)
FLOOR = 1
BLOCKED = 2
TRAP = 3
FOOD = 4
PLAYER = 5
ENEMY = 9


def startGameWindow():
    screen.fill(BACKGROUND_COLOR)
    pygame.display.set_caption("Catch me if you can")
    game_icon = pygame.image.load('player_down.png')
    pygame.display.set_icon(game_icon)


class Player(object):
    def __init__(self):
        self.x = 500  # random.randrange(32, 600, 32)
        self.y = 500
        GAME_WORLD[500][500] = PLAYER  # value of player is 5
        self.current_tile_val = 1
        self.health = 100
        self.score = 0
        self.previous_dir = "left"

    def move(self, direction=""):
        GAME_WORLD[self.x][self.y] = self.current_tile_val
        current_pos_x = self.x
        current_pos_y = self.y

        if direction == "":
            direction = self.previous_dir

        if direction == "left":
            self.x = self.x - 1
        elif direction == "right":
            self.x = self.x + 1
        elif direction == "up":
            self.y = self.y - 1
        elif direction == "down":
            self.y = self.y + 1

        self.previous_dir = direction

        if GAME_WORLD[self.x][self.y] == BLOCKED:
            self.x = current_pos_x
            self.y = current_pos_y

        self.current_tile_val = GAME_WORLD[self.x][self.y]
        GAME_WORLD[self.x][self.y] = PLAYER
        point = self.update_health()
        self.health = min(100, self.health + point)

    def update_health(self):
        if self.current_tile_val == FLOOR:
            return -1
        if self.current_tile_val == TRAP:
            self.current_tile_val = FLOOR
            return -10
        if self.current_tile_val == FOOD:
            self.current_tile_val = FLOOR
            self.score = self.score + 10
            return 20
        if self.current_tile_val == ENEMY:
            self.health = 0
            return -1


class Enemy(object):
    def __init__(self):
        # spawn near the Player
        xx = np.random.randint(-4, 5)
        yy = np.random.randint(-4, 5)
        if xx == 0 or yy == 0:
            xx = xx + 1
            yy = yy + 1
        self.x = 500 + xx
        self.y = 500 + yy
        GAME_WORLD[self.x][self.y] = ENEMY
        self.current_tile_val = 1
        self.health = 100
        self.reward = 0
        self.previous_dir = "left"

    def move(self, direction=""):
        GAME_WORLD[self.x][self.y] = self.current_tile_val
        current_pos_x = self.x
        current_pos_y = self.y

        if direction == "":
            direction = self.previous_dir

        if direction == "left":
            self.x = self.x - 1
        elif direction == "right":
            self.x = self.x + 1
        elif direction == "up":
            self.y = self.y - 1
        elif direction == "down":
            self.y = self.y + 1

        self.previous_dir = direction

        if GAME_WORLD[self.x][self.y] == BLOCKED:
            self.x = current_pos_x
            self.y = current_pos_y

        self.current_tile_val = GAME_WORLD[self.x][self.y]
        GAME_WORLD[self.x][self.y] = ENEMY
        self.reward = self.reward + self.get_reward()

    def get_reward(self):
        if self.current_tile_val == FLOOR:
            self.health = self.health - 1
            return -2
        if self.current_tile_val == TRAP:
            self.current_tile_val = FLOOR
            self.health = self.health - 10
            return -20
        if self.current_tile_val == FOOD:
            self.current_tile_val = FLOOR
            self.health = self.health + 2
            return 10
        if self.current_tile_val == PLAYER:
            self.health = 0
            return 200


def updateGameFrame(player):
    start_x = player.x - FRAME_SIZE
    start_y = player.y - FRAME_SIZE
    end_x = player.x + FRAME_SIZE
    end_y = player.y + FRAME_SIZE
    return (start_x, end_x), (start_y, end_y)


class Game(object):
    def __init__(self):
        self.player = Player()
        self.enemy = Enemy()
        self.score = 0
        self.reward = 0
        self.frame_x, self.frame_y = updateGameFrame(self.player)
        self.GAME_FRAME = GAME_WORLD[self.frame_x[0]:self.frame_x[1]+1, self.frame_y[0]:self.frame_y[1]+1]

    def processEvents(self):
        moved = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            # Get keyboard input and move player accordingly
            elif event.type == pygame.KEYDOWN:
                moved = True
                if event.key == pygame.K_LEFT:
                    self.player.move("left")
                    self.enemy.move("left")
                elif event.key == pygame.K_RIGHT:
                    self.player.move("right")
                    self.enemy.move("right")
                elif event.key == pygame.K_UP:
                    self.player.move("up")
                    self.enemy.move("up")
                elif event.key == pygame.K_DOWN:
                    self.player.move("down")
                    self.enemy.move("down")

        if not moved:
            self.player.move()
            self.enemy.move()
        if self.player.health <= 0 or self.enemy.health <= 0:
            return True
        return False

    def draw(self):
        self.frame_x, self.frame_y = updateGameFrame(self.player)
        self.GAME_FRAME = GAME_WORLD[self.frame_x[0]:self.frame_x[1]+1, self.frame_y[0]:self.frame_y[1]+1]

        for x in range(0, FRAME_UNIT):
            for y in range(0, FRAME_UNIT):
                val = self.GAME_FRAME[x][y]
                tile_img = None

                if val == FLOOR:
                    tile_img = img_floor
                elif val == BLOCKED:
                    tile_img = img_block
                elif val == TRAP:
                    tile_img = img_trap
                elif val == FOOD:
                    tile_img = img_food
                elif val == PLAYER:
                    if self.player.previous_dir == "left":
                        tile_img = img_player_left
                    elif self.player.previous_dir == "right":
                        tile_img = img_player_right
                    elif self.player.previous_dir == "up":
                        tile_img = img_player_up
                    elif self.player.previous_dir == "down":
                        tile_img = img_player_down
                elif val == ENEMY:
                    if self.enemy.previous_dir == "left":
                        tile_img = img_enemy_left
                    elif self.enemy.previous_dir == "right":
                        tile_img = img_enemy_right
                    elif self.enemy.previous_dir == "up":
                        tile_img = img_enemy_up
                    elif self.enemy.previous_dir == "down":
                        tile_img = img_enemy_down

                screen.blit(tile_img, (PIXEL_UNIT * x, PIXEL_UNIT * y))

        text_color = (0, 0, 255)
        player_health = font.render("Health : " + str(self.player.health), True, text_color)
        screen.blit(player_health, (2, 2))
        player_score = font.render("Score : " + str(self.player.score), True, text_color)
        screen.blit(player_score, (2, 22))

        enemy_health = font.render("Health : " + str(self.enemy.health), True, text_color)
        screen.blit(enemy_health, (SCREEN_WIDTH - 100, 2))
        enemy_reward = font.render("Reward : " + str(self.enemy.reward), True, text_color)
        screen.blit(enemy_reward, (SCREEN_WIDTH - 100, 22))

def main():
    clock = pygame.time.Clock()
    startGameWindow()
    game = Game()
    done = False
    while not done:
        done = game.processEvents()
        game.draw()
        pygame.display.update()
        clock.tick(3)
        if done:
            pygame.time.delay(3000)


main()
