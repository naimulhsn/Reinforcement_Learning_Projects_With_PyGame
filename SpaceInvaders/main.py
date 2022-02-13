import pygame
import random

from pygame import mixer

pygame.init()

screen = pygame.display.set_mode((800, 600))

font = pygame.font.Font('freesansbold.ttf', 14)

# Background Sound
mixer.music.load('background.wav')
mixer.music.set_volume(.2)
mixer.music.play(-1)

background = pygame.image.load('bg2.jpg')

# Render Window
pygame.display.set_caption("Space Invaders")
gameIcon = pygame.image.load('spaceship.png')
pygame.display.set_icon(gameIcon)

# Score
score = 0
# Player
playerImg = pygame.image.load('spaceship_64.png')
playerX = 370
playerY = 480
playerX_change = 0
x_change = 1

# Bullet
bulletImg = pygame.image.load('bullet_24.png')
bulletX = 400
bulletY = 500
bulletY_change = 1
bullet_state = "ready"

# Enemy
enemyImg = pygame.image.load('ufo_64.png')
enemyX = random.randint(0, 736)
enemyY = random.randrange(50, 151, 50)  # [start, start+step, ..., stop-1]
enemyX_change = 0.8
enemyY_change = 50

enemyDestroyImg = pygame.image.load('ufo_64_destroyed.png')
destroyedX = 0
destroyedY = 0
destroyed = 0


def player(x, y):
    screen.blit(playerImg, (x, y))


def enemy(x, y):
    screen.blit(enemyImg, (x, y))


def fire_bullet(x, y):
    global bullet_state
    bullet_state = "fire"
    screen.blit(bulletImg, (x + 20, y))


def bullet_hits(bulletX, bulletY, enemyX, enemyY):
    if (enemyY + 45) > bulletY >= (enemyY - 5) and (enemyX + 32) > bulletX >= (enemyX - 32):
        return True
    return False


def show_blast(x, y, d):
    screen.blit(enemyDestroyImg, (x, y))


def enemy_hits(playerX, playerY, enemyX, enemyY):
    if (enemyY + 50) > playerY >= (enemyY - 50) and (enemyX + 64) > playerX >= (enemyX - 55):
        return True
    return False


# Game Loop
running = True
while running:
    # Game Frame
    screen.fill((0, 0, 0))
    screen.blit(background, (0, 0))
    # For Keyboard Events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            # pygame.quit()

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                playerX_change = -x_change
            if event.key == pygame.K_RIGHT:
                playerX_change = x_change
            if event.key == pygame.K_SPACE:
                if bullet_state == "ready":
                    bullet_state = "fire"
                    bulletX = playerX
                    bulletY = playerY - 20
                    bullet_sound = mixer.Sound('laser.wav')
                    bullet_sound.play()

        if event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                playerX_change = 0

    playerX += playerX_change
    playerX = max(0, min(736, playerX))
    player(playerX, playerY)

    enemyX += enemyX_change
    if enemyX <= 0 or enemyX >= 736:
        enemyX_change *= -1
        enemyY += enemyY_change
    enemy(enemyX, enemyY)

    if bullet_state == "fire":
        fire_bullet(bulletX, bulletY)
        bulletY -= bulletY_change
        if bulletY <= 0:
            bullet_state = "ready"
        if bullet_hits(bulletX, bulletY, enemyX, enemyY):
            bullet_state = "ready"
            destroyed = 128
            destroyedX = enemyX
            destroyedY = enemyY
            enemyX = random.randint(0, 736)
            enemyY = random.randint(40, 150)
            enemyX_change *= -1
            score += 1
            explosion_sound = mixer.Sound('explosion.wav')
            explosion_sound.set_volume(.2)
            explosion_sound.play()

    if destroyed > 0:
        show_blast(destroyedX, destroyedY, destroyed)
        destroyed -= 2

    if enemy_hits(playerX, playerY, enemyX, enemyY):
        enemyX = random.randint(0, 736)
        enemyY = random.randint(40, 150)
        enemyX_change *= -1
        score = 0
        playerX = 370
        playerY = 480
        explosion_sound = mixer.Sound('explosion.wav')
        explosion_sound.set_volume(.2)
        explosion_sound.play()

    text = font.render("Score : " + str(score), True, (255, 255, 255))
    screen.blit(text, (15, 15))

    pygame.display.update()
