import pygame
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import adam_v2
from collections import deque
import random
from tqdm import tqdm
import matplotlib.pyplot as plt  # for graphing our mean rewards over time

DISCOUNT = 0.9
REPLAY_MEMORY_SIZE = 10000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 300  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = 'model'
# Environment settings
EPISODES = 2000
# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.999
MIN_EPSILON = 0.001

SHOW_EVERY = 100
PLOT_EVERY = 10

SPAWN_POINT = 125
FRAME_SIZE = 5  # Actual frame size is Frame_size*2+1
FRAME_UNIT = (FRAME_SIZE * 2 + 1)
PIXEL_UNIT = 32
SCREEN_WIDTH = FRAME_UNIT * PIXEL_UNIT  # 32 * 20 (20 = col)
SCREEN_HEIGHT = FRAME_UNIT * PIXEL_UNIT  # 32 * 15 (15 = row)
pygame.init()
screen = None

GAME_WORLD = np.random.randint(1, 12, (250, 250))
GAME_WORLD = np.where(GAME_WORLD > 4, 1, GAME_WORLD)

FLOOR = 1
BLOCKED = 2
TRAP = 3
FOOD = 4
PLAYER = 7
ENEMY = 9


class Player(object):
    def __init__(self):
        self.x = SPAWN_POINT  # random.randrange(32, 600, 32)
        self.y = SPAWN_POINT
        GAME_WORLD[SPAWN_POINT][SPAWN_POINT] = PLAYER  # value of player is 5
        self.current_tile_val = 1
        self.prev_dir = 0

    def move(self, direction=-1):
        if GAME_WORLD[self.x][self.y] == ENEMY:
            return True, -100000
        GAME_WORLD[self.x][self.y] = self.current_tile_val
        current_pos_x = self.x
        current_pos_y = self.y

        if np.random.randint(1, 10) < 5:
            direction = np.random.randint(0, 4)
            self.prev_dir = direction
        else:
            direction = self.prev_dir

        if direction == 0:  # left
            self.x = self.x - 1
        elif direction == 1:  # right
            self.x = self.x + 1
        elif direction == 2:  # up
            self.y = self.y - 1
        elif direction == 3:  # down
            self.y = self.y + 1

        if GAME_WORLD[self.x][self.y] == BLOCKED:
            self.x = current_pos_x
            self.y = current_pos_y
        elif GAME_WORLD[self.x][self.y] == ENEMY:
            return True, 100  # if player runs into enemy

        self.current_tile_val = GAME_WORLD[self.x][self.y]
        GAME_WORLD[self.x][self.y] = PLAYER
        return False, -100000


class Enemy(object):
    MOVE_PENALTY = -2
    TRAP_PENALTY = -20
    FOOD_REWARD = 10
    CATCHING_REWARD = 200

    def __init__(self):
        # spawn near the Player
        xx = np.random.randint(-4, 5)
        yy = np.random.randint(-4, 5)
        while -2 < xx < 2 and -2 < yy < 2:
            xx = np.random.randint(-4, 5)
            yy = np.random.randint(-4, 5)
        self.x = SPAWN_POINT + xx
        self.y = SPAWN_POINT + yy
        GAME_WORLD[self.x][self.y] = ENEMY
        self.current_tile_val = 1

    def move(self, direction):
        GAME_WORLD[self.x][self.y] = self.current_tile_val
        current_pos_x = self.x
        current_pos_y = self.y

        if direction == 0:
            self.x = self.x - 1
        elif direction == 1:
            self.x = self.x + 1
        elif direction == 2:
            self.y = self.y - 1
        elif direction == 3:
            self.y = self.y + 1

        if GAME_WORLD[self.x][self.y] == BLOCKED:
            self.x = current_pos_x
            self.y = current_pos_y

        self.current_tile_val = GAME_WORLD[self.x][self.y]
        GAME_WORLD[self.x][self.y] = ENEMY
        return self.get_reward()

    def get_reward(self):
        if self.current_tile_val == FLOOR:
            return self.MOVE_PENALTY
        if self.current_tile_val == TRAP:
            self.current_tile_val = FLOOR
            return self.TRAP_PENALTY
        if self.current_tile_val == FOOD:
            self.current_tile_val = FLOOR
            return self.FOOD_REWARD
        if self.current_tile_val == PLAYER:
            return self.CATCHING_REWARD


def start_game_window():
    global screen
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    screen.fill((0, 0, 0))
    pygame.display.set_caption("Catch me if you can")


def observation(player):
    start_x = player.x - FRAME_SIZE
    start_y = player.y - FRAME_SIZE
    end_x = player.x + FRAME_SIZE
    end_y = player.y + FRAME_SIZE
    # return (start_x, end_x), (start_y, end_y)
    return GAME_WORLD[start_x: end_x + 1, start_y: end_y + 1]


class GameEnv(object):
    OBSERVATION_SPACE_VALUES = (FRAME_UNIT, FRAME_UNIT, 1)  # 4
    ACTION_SPACE_SIZE = 4

    img_floor = pygame.image.load("floor.png")
    img_block = pygame.image.load("block.png")
    img_trap = pygame.image.load("trap.png")
    img_food = pygame.image.load("food.png")
    img_player_down = pygame.image.load("player_down.png")
    img_enemy_down = pygame.image.load("enemy_down.png")

    def reset(self):
        global GAME_WORLD
        GAME_WORLD = np.random.randint(1, 12, (250, 250))
        GAME_WORLD = np.where(GAME_WORLD > 4, 1, GAME_WORLD)
        self.player = Player()
        self.enemy = Enemy()
        self.episode_step = 0
        return observation(self.enemy)

    def step(self, action):
        self.episode_step += 1
        reward = self.enemy.move(action)
        done, reward_p = self.player.move()
        reward = max(reward, reward_p)
        new_observation = observation(self.enemy)
        return new_observation, reward, done

    def clear(self):
        GAME_WORLD[self.player.x][self.player.y] = 1
        GAME_WORLD[self.enemy.x][self.enemy.y] = 1

    def render(self):
        GAME_FRAME = observation(self.player)

        for x in range(0, FRAME_UNIT):
            for y in range(0, FRAME_UNIT):
                val = GAME_FRAME[x][y]
                tile_img = None

                if val == FLOOR:
                    tile_img = self.img_floor
                elif val == BLOCKED:
                    tile_img = self.img_block
                elif val == TRAP:
                    tile_img = self.img_trap
                elif val == FOOD:
                    tile_img = self.img_food
                elif val == PLAYER:
                    tile_img = self.img_player_down
                elif val == ENEMY:
                    tile_img = self.img_enemy_down

                screen.blit(tile_img, (PIXEL_UNIT * x, PIXEL_UNIT * y))
        pygame.display.update()


env = GameEnv()

np.random.seed(1)


# Agent class
class DQNAgent:
    def __init__(self):

        # Main model
        self.model = self.create_model()
        self.model.load_weights(f'model_weight/{MODEL_NAME}.h5')

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()

        model.add(
            Conv2D(256, (3, 3), input_shape=env.OBSERVATION_SPACE_VALUES))  # OBSERVATION_SPACE_VALUES = (11, 11, 1)
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.1))

        model.add(Conv2D(128, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.1))

        model.add(Flatten())
        model.add(Dense(32))

        model.add(Dense(env.ACTION_SPACE_SIZE, activation='linear'))  # ACTION_SPACE_SIZE = how many choices (4)
        model.compile(loss="mse", optimizer=adam_v2.Adam(learning_rate=0.01), metrics=['accuracy'])
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch]) / 10
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch]) / 10
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X) / 10, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape) / 10)[0]


agent = DQNAgent()
reward_avg = 0
episode_rewards = []
# Iterate over episodes
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

    if episode % SHOW_EVERY == 0:
        start_game_window()
    # Update tensorboard step every episode

    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1

    # Reset environment and get initial state
    current_state = env.reset()

    # Reset flag and start iterating until episode ends
    done = False
    while not done:
        # This part stays mostly the same, the change is to query a model for Q values
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(agent.get_qs(current_state))
        else:
            # Get random action
            action = np.random.randint(0, env.ACTION_SPACE_SIZE)

        new_state, reward, done = env.step(action)

        # Transform new continous state to new discrete state and count reward
        episode_reward += reward

        if episode % SHOW_EVERY == 0:
            env.render()

        # Every step we update replay memory and train main network
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done)

        current_state = new_state
        step += 1
        if step > 110:
            done = True

    reward_avg += episode_reward
    if episode % PLOT_EVERY == 0:
        episode_rewards.append(reward_avg / PLOT_EVERY)
        reward_avg = 0
    # Decay epsilon

    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)
    env.clear()
    if episode % SHOW_EVERY == 0:
        pygame.quit()
        agent.model.save(f'model_weight/{MODEL_NAME}___{sum(episode_rewards[-10:])/SHOW_EVERY}__.h5')

# agent.model.save(f'models/{MODEL_NAME}.model')
agent.model.save(f'model_weight/{MODEL_NAME}_final.h5')
plt.plot([i for i in range(len(episode_rewards))], episode_rewards)
plt.ylabel("Reward")
plt.xlabel("episode #")
plt.show()
