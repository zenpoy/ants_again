import argparse
import random
import numpy as np
import pygame
import os
from sklearn.neighbors import NearestNeighbors
import itertools

ANT_SIZE = 5
FOOD_SIZE = 10

def relu(x):
    return (abs(x) + x) / 2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

class Database(object):
    def __init__(self):
        self.points = np.empty(shape=(0, 2))

    def insert_position(self, x, y):
        self.points = np.concatenate([self.points, np.array([[x, y]])])
        return self.points.shape[0] - 1

    def set_position(self, index, x, y):
        assert index < self.points.shape[0]
        self.points[index, :] = np.array((x, y))

    def get_position(self, index):
        assert index < self.points.shape[0]
        return self.points[index, :]


class Ant(pygame.sprite.Sprite):
    def __init__(self, x, y, ant_width, ant_height, color, world_width, world_height, layers, db):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface([ant_width, ant_height])
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.db = db
        self.index = self.db.insert_position(x, y)
        self.vx = 0
        self.vy = 0
        self.v_decay = 0.5
        self.world_width = world_width
        self.world_height = world_height
        self.radius = 0.1

        self.layers = layers # each layer is a matrix and bias, last layer is embedding into output and sigmoid activation

    def get_layers(self):
        return self.layers

        
    def update(self, events, dt, world):
        dt = dt / 10000.0

        output = np.array([[0.1, 0.1]])

        x, y = self.db.get_position(self.index)

        input_data = world.sense(x, y, self.layers[0][0].shape[0] // 2, self.radius).flatten()

        output = input_data
        total_layers = len(self.layers)
        for i, (weights, bias) in enumerate(self.layers):
            if i < (total_layers - 1):
                output = relu((np.dot(output, weights) + bias))
            else:
                output = tanh((np.dot(output, weights) + bias))


        self.vx = output[0]
        self.vy = output[1]
        x = (x + self.vx * dt)
        y = (y + self.vy * dt)
        self.rect.centerx = x * self.world_width
        self.rect.centery = y * self.world_height

        self.db.set_position(self.index, x, y)
    

class Food(pygame.sprite.Sprite):
    def __init__(self, x, y, amount, food_width, food_height, color, world_width, world_height, db):
        pygame.sprite.Sprite.__init__(self)
        self.db = db
        self.index = self.db.insert_position(x, y)
        self.amount = amount
        self.world_width = world_width
        self.world_height = world_height

        self.image = pygame.Surface([food_width, food_height])
        self.image.fill(color)
        self.rect = self.image.get_rect()

        self.rect.centerx = x * self.world_width
        self.rect.centery = y * self.world_height


class World(object):
    def __init__(self, db):
        self.ants = pygame.sprite.Group()
        self.foods = pygame.sprite.Group()
        self.db = db

    def add_ant(self, ant):
        self.ants.add(ant)

    def add_food(self, food):
        self.foods.add(food)

    def sense(self, x, y, k, radius):
        nbrs = NearestNeighbors(
            n_neighbors=k, algorithm='ball_tree').fit(self.db.points)
        knn = nbrs.kneighbors(X=[[x, y]])
        knn_points = self.db.points[knn[1][knn[0] < radius]]

        res = np.ones(shape=[k, 2]) * -1
        res[:knn_points.shape[0], :] = knn_points

        return res.T

    def score(self):
        nbrs = NearestNeighbors(
            n_neighbors=2, algorithm='ball_tree').fit(self.db.points)

        max_distance = 10.0
        distance = max_distance
        for food in self.foods:
            distances, indices = nbrs.kneighbors(X=[self.db.get_position(food.index)])
            for d, i in zip(distances[0], indices[0]):
                if i != food.index:
                    distance = d

        return max(max_distance - d, 0)

# class Game(object):
#     def __init__(self):
        

# def create_games(num_of_games):


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Ants, again")
    parser.add_argument("--num_ants", type=int, default=100)
    parser.add_argument("--num_food", type=int, default=1)
    parser.add_argument("--max_food", type=int, default=1000)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--steps_per_epoch", type=int, default=100)
    parser.add_argument("--num_of_games", type=int, default=10)

    args = parser.parse_args()

    pygame.init()

    size = width, height = args.width, args.height

    screen = pygame.display.set_mode(size)
    myfont = pygame.font.SysFont("mono", 16)

    # games = create_games(args.num_of_games)

    # while True:
    #     scores = []
        
    #     for game in games:
    #         game.play(args.steps_per_epoch)
        
    #     games = sorted(games, key=lambda x: x.score(), reverse=True)

    #     num_new_games = args.num_of_games // 3
    #     num_mutated_games = num_new_games
    #     num_best_games = args.num_of_games - \
    #         (num_new_games + num_mutated_games)
        
    #     games = create_games(num_new_games) + \
    #         mutate_games(games[:num_mutated_games]) + \
    #         copy_games(games[:num_best_games])


    score = 0

    db = Database()

    world = World(db)

    area = 9
    hidden1 = 10

    for i in range(args.num_ants):
        layers = [(np.random.random(size=[area * 2, hidden1]) - 0.5, np.random.random(size=[hidden1]) -
                   0.5), (np.random.random(size=[hidden1, 2]) - 0.5, np.random.random(size=[2]) - 0.5)]
        world.add_ant(Ant(x=random.random(), y=random.random(), ant_width=ANT_SIZE, ant_height=ANT_SIZE, color=[0,0,0], world_width=width, world_height=height, layers=layers, db=db))

    for i in range(args.num_food):
        world.add_food(Food(x=random.random(), y=random.random(), amount=random.random() * args.max_food, food_width=FOOD_SIZE, food_height=FOOD_SIZE, color=[
                       255, 0, 0], world_width=width, world_height=height, db=db))


    clock = pygame.time.Clock()
    dt = 0
    while True:
        events = pygame.event.get()
        for e in events:
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_q:
                    exit()
                if e.key == pygame.K_ESCAPE:
                    exit()

            if e.type == pygame.QUIT:
                exit()

        world.ants.update(events, dt, world)

        # world.update()

        score = world.score()

        screen.fill((255, 255, 255))
        scoretext = myfont.render("Score {0:7.2f}".format(score), 1, (0, 0, 0))
        screen.blit(scoretext, (5, 10))
        world.ants.draw(screen)
        world.foods.draw(screen)
        pygame.display.update()

        dt = clock.tick(60)



    
