import argparse
import random
import numpy as np
import pygame
import os
from sklearn.neighbors import NearestNeighbors
import itertools
import copy

ANT_SIZE = 5
FOOD_SIZE = 10
FOOD_POS = 0.5, 0.5
ANT_POS = 0.1, 0.1

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
    def __init__(self, index, ant_width, ant_height, color, world_width, world_height, layers):
        pygame.sprite.Sprite.__init__(self)
        self.ant_width = ant_width
        self.ant_height = ant_height
        self.color = color
        self.image = pygame.Surface([ant_width, ant_height])
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.index = index
        self.vx = 0
        self.vy = 0
        self.v_decay = 0.5
        self.world_width = world_width
        self.world_height = world_height
        self.radius = 0.1
        self.layers = layers # each layer is a matrix and bias, last layer is embedding into output and sigmoid activation
        self.mutation_factor = 9

    @classmethod
    def fromother(cls, ant):
        return cls(ant.index, ant.ant_width, ant.ant_height, ant.color, ant.world_width, ant.world_height, copy.deepcopy(ant.layers))

    def get_layers(self):
        return self.layers

    def mutate(self):
        for i, (weights, bias) in enumerate(self.layers):
            self.layers[i] = weights + np.random.normal(
                size=weights.shape, scale=self.mutation_factor), bias + np.random.normal(size=bias.shape)


        
    def update(self, events, dt, world):
        dt = dt / 100.0

        output = np.array([[0.1, 0.1]])

        x, y = world.db.get_position(self.index)

        input_data = world.sense(x, y, self.layers[0][0].shape[0] // 2, self.radius).flatten()

        output = input_data
        total_layers = len(self.layers)
        for i, (weights, bias) in enumerate(self.layers):
            if i < (total_layers - 1):
                output = sigmoid((np.dot(output, weights) + bias))
            else:
                output = tanh((np.dot(output, weights) + bias))


        self.vx = output[0]
        self.vy = output[1]
        x = (x + self.vx * dt)
        y = (y + self.vy * dt)
        self.rect.centerx = x * self.world_width
        self.rect.centery = y * self.world_height

        world.db.set_position(self.index, x, y)
    

class Food(pygame.sprite.Sprite):
    def __init__(self, index, amount, food_width, food_height, color, world_width, world_height):
        pygame.sprite.Sprite.__init__(self)
        self.index = index
        self.amount = amount

        # read only
        self.food_width = food_width
        self.food_height = food_height
        self.color = color

        self.world_width = world_width
        self.world_height = world_height

        self.image = pygame.Surface([food_width, food_height])
        self.image.fill(color)
        self.rect = self.image.get_rect()

    @classmethod
    def fromother(cls, food):
        return cls(food.index, food.amount, food.food_width, food.food_height, food.color, food.world_width, food.world_height)


    def update(self, events, dt, world):
        x, y = world.db.get_position(self.index)

        self.rect.centerx = x * self.world_width
        self.rect.centery = y * self.world_height

        # world.db.set_position(self.index, x, y)

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
        res[:knn_points.shape[0], :] = knn_points - np.array([x, y])

        return res.T

    def score(self):
        k = len(self.ants)
        nbrs = NearestNeighbors(
            n_neighbors=k, algorithm='ball_tree').fit(self.db.points)

        total_distances = 0
        for food in self.foods:
            distances, indices = nbrs.kneighbors(X=[self.db.get_position(food.index)])
            total_distances += np.sum(distances)

        
        return total_distances / (k * len(self.foods))

class Game(object):
    def __init__(self, receptive_area, max_food):
        self.db = Database()
        self.world = World(self.db)
        self.receptive_area = receptive_area
        self.hidden = 50
        self.max_food = max_food
        self.score = 0
        self.steps = 0


    def populate_world_randomly(self,num_ants, num_food):
        layers = [(np.random.random(size=[self.receptive_area * 2, self.hidden]) - 0.5, np.random.random(size=[self.hidden]) -
                   0.5), (np.random.random(size=[self.hidden, 2]) - 0.5, np.random.random(size=[2]) - 0.5)]
        for i in range(num_ants):
            x = ANT_POS[0]
            y = ANT_POS[1]
            assigned_index = self.db.insert_position(x, y)
            self.world.add_ant(Ant(index=assigned_index, ant_width=ANT_SIZE, ant_height=ANT_SIZE, color=[
                        0, 0, 0], world_width=width, world_height=height, layers=layers.copy()))

        for i in range(num_food):
            x = FOOD_POS[0]  
            y = FOOD_POS[1] 
            assigned_index = self.db.insert_position(x, y)
            self.world.add_food(Food(index=assigned_index, amount=random.random() * self.max_food, food_width=FOOD_SIZE, food_height=FOOD_SIZE, color=[
                        255, 0, 0], world_width=width, world_height=height))

    def populate_world_from_other(self, other):
        for i, ant in enumerate(other.world.ants):
            x = ANT_POS[0]
            y = ANT_POS[1]
            assigned_index = self.db.insert_position(x, y)
            # assigned_index = self.db.insert_position(*other.db.get_position(ant.index))
            new_ant = Ant.fromother(ant)
            new_ant.index = assigned_index
            self.world.add_ant(new_ant)
        
        for i, food in enumerate(other.world.foods):
            x = FOOD_POS[0]
            y = FOOD_POS[1]
            assigned_index= self.db.insert_position(x, y)
            new_food = Food.fromother(food)
            new_food.index = assigned_index
            self.world.add_food(new_food)


    def copy(self):
        copied_game = Game(self.receptive_area, self.max_food)
        copied_game.populate_world_from_other(self)
        return copied_game

    def mutate(self):
        mutated_game = self.copy()
        for ant in mutated_game.world.ants:
            ant.mutate()
        return mutated_game
        


    def play(self, steps, generation, index):
        dt = 1
        for step in range(steps):
            self.steps += 1
            events = pygame.event.get()
            for e in events:
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_q:
                        exit()
                    if e.key == pygame.K_ESCAPE:
                        exit()

                if e.type == pygame.QUIT:
                    exit()

            self.world.ants.update(events, dt, self.world)
            self.world.foods.update(events, dt, self.world)

            # world.update()

            self.score += self.world.score()

            screen.fill((255, 255, 255))
            scoretext = myfont.render(
                "Game - {2} - Generation: {1} - Score {0:7.2f}".format(self.get_score(), generation, index), 1, (0, 0, 0))
            screen.blit(scoretext, (5, 10))
            self.world.ants.draw(screen)
            self.world.foods.draw(screen)
            pygame.display.update()

            # dt = clock.tick(60)

    def get_score(self):
        return self.score / (self.steps + 1)

        

def create_games(args):
    new_games = [Game(args.receptive_area, args.max_food) for x in range(args.num_of_games)]
    for game in new_games:
        game.populate_world_randomly(args.num_ants, args.num_food)
    return new_games

def mutate_games(games):
    return [game.mutate() for game in games]


def copy_games(games):
    return [game.copy() for game in games]


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Ants, again")
    parser.add_argument("--num_ants", type=int, default=10)
    parser.add_argument("--num_food", type=int, default=1)
    parser.add_argument("--max_food", type=int, default=1000)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--steps_per_epoch", type=int, default=25)
    parser.add_argument("--num_of_games", type=int, default=5)
    parser.add_argument("--receptive_area", type=int, default=9)

    args = parser.parse_args()

    pygame.init()

    size = width, height = args.width, args.height

    screen = pygame.display.set_mode(size)
    myfont = pygame.font.SysFont("mono", 16)

    score = 0
    generation = 0

    games = create_games(args)
    
    while True:
        generation += 1
        for i, game in enumerate(games):
            game.play(args.steps_per_epoch, generation, i)

        games = sorted(games, key=lambda x: x.get_score(), reverse=False)

        print("Generation {}: Best score: {}".format(
            generation, games[0].get_score()))
    
        num_mutated_games = args.num_of_games // 2
        num_best_games = args.num_of_games - num_mutated_games

        mutated_games = copy_games(games[:num_best_games]) + \
            mutate_games(games[:num_mutated_games])

        games = mutated_games



    
