import argparse
import random
import numpy as np
import pygame
import os
from sklearn.neighbors import NearestNeighbors
import itertools
import copy
from scipy.spatial import distance
from typing import Any
from scipy.special import expit

THINGS_TO_SENSE = 3
ANT_SIZE = 5
FOOD_SIZE = 10
FOOD_POS = 0.5, 0.5
ANT_POS = 0.5, 0.5
ANT_RADIUS = 0.4
INPUT_SIZE = 2 + THINGS_TO_SENSE
OUTPUT_SIZE = 3
RADIUS = 0.5
MUTATION_FACTOR = 0.025


def rand_pos(center, radius):
    rand_theta = random.random() * np.pi * 2
    ant_pos = center[0] + np.cos(rand_theta) * radius,  center[1] + np.sin(rand_theta) * radius
    return ant_pos


def sliding_window(iterable, n=2):
    iterables = itertools.tee(iterable, n)

    for iterable, num_skipped in zip(iterables, itertools.count()):
        for _ in range(num_skipped):
            next(iterable, None)

    return zip(*iterables)

def relu(x):
    return (abs(x) + x) / 2

def sigmoid(x):
    return np.where(-x > np.log(np.finfo(x.dtype).max), 0.0, 1 / (1 + expit(-x)))


# def sigmoid(x):
#     res = np.where(-x > np.log(np.finfo(x.dtype).max), 0.0, np.where(x >=
#                    0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x))))
#     return res

def tanh(x):
    return np.tanh(x)

def to_one_hot(a, num_cols):
    b = np.zeros((a.size, num_cols))
    b[np.arange(a.size), a] = 1
    return b

class DecayDatabase(object):
    def __init__(self):
        self.points = np.empty(shape=(0, 2))
        self.values = np.empty(shape=(0, 1))
        self.decay_factor = 0.99
        self.threshold = 0.1

    def insert_position(self, x, y):
        self.points = np.concatenate([self.points, np.array([[x, y]])])
        self.values = np.concatenate([self.values, np.array([[1.0]])])

    def update(self):
        self.values *= self.decay_factor
        mask = (self.values > self.threshold).flatten()
        self.points = self.points[mask, :]
        self.values = self.values[mask, :]



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
    def __init__(self, index, ant_width, ant_height, color, world_width, world_height, layers, receptive_area):
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
        self.radius = RADIUS
        self.layers = layers # each layer is a matrix and bias, last layer is embedding into output and sigmoid activation
        self.mutation_factor = MUTATION_FACTOR
        self.receptive_area = receptive_area
        self.score = 0

    @classmethod
    def fromother(cls, ant):
        new_ant = cls(ant.index, ant.ant_width, ant.ant_height, ant.color, ant.world_width,
                      ant.world_height, copy.deepcopy(ant.layers), ant.receptive_area)
        new_ant.score = ant.score
        return new_ant

    def mutate(self):
        for i, (weights, bias) in enumerate(self.layers):
            self.layers[i] = weights + np.multiply(np.random.normal(
                size=weights.shape, scale=self.mutation_factor), np.random.normal(
                size=weights.shape) > 0.25), bias + np.multiply(np.random.normal(size=bias.shape, scale=self.mutation_factor), np.random.normal(size=bias.shape) > 0.25)

    def get_score(self):
        return np.average(self.score)


    @staticmethod
    def _decode_output(output):
        vx = output[0] 
        vy = output[1] 
        is_trace = output[2] > 0

        return vx, vy, is_trace
        
    def update(self, events : Any, dt : float, world : 'World') -> None:
        dt = dt / 10.0

        output = np.array([[0.1, 0.1]])

        x, y = world.ant_db.get_position(self.index)

        input_data = world.sense(
            x, y, self.receptive_area, self.radius)
        

        output = input_data
        total_layers = len(self.layers)

        for i in range(total_layers - 2):
            weights, bias = self.layers[i]
            output = relu((np.dot(output, weights) + bias))

        local_features = output

        global_features = local_features

        for i in range(total_layers - 2, total_layers - 1):
            weights, bias = self.layers[i]
            global_features = relu(
                (np.dot(global_features, weights) + bias))

        global_features = np.max(global_features, axis=0)

        global_features = np.array([global_features] * local_features.shape[0])

        weights, bias = self.layers[total_layers - 1]
        
        output = np.concatenate([local_features, global_features], axis=1)
        
        output = tanh((np.dot(output, weights) + bias))
        # output = sigmoid((np.dot(output, weights) + bias))
        # output = (np.dot(output, weights) + bias)

        output = np.max(output, axis=0)

        vx, vy, is_trace = self._decode_output(output)


        self.vx = vx
        self.vy = vy
        old_x = x
        old_y = y
        x = (x + self.vx * dt)
        y = (y + self.vy * dt)
        self.rect.centerx = x * self.world_width
        self.rect.centery = y * self.world_height

        world.ant_db.set_position(self.index, x, y)
        if is_trace:
            world.trace_db.insert_position(old_x, old_y)
    

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
        x, y = world.food_db.get_position(self.index)

        self.rect.centerx = x * self.world_width
        self.rect.centery = y * self.world_height

        # world.db.set_position(self.index, x, y)

class World(object):
    def __init__(self):
        self.ants = pygame.sprite.Group()
        self.foods = pygame.sprite.Group()
        self.ant_db = Database()
        self.trace_db = DecayDatabase()
        self.food_db = Database()

    def add_ant(self, ant, pos):
        x, y = pos
        assigned_index = self.ant_db.insert_position(x, y)
        ant.index = assigned_index
        self.ants.add(ant)

    def add_food(self, food, pos):
        x, y = pos
        assigned_index = self.food_db.insert_position(x, y)
        food.index = assigned_index
        self.foods.add(food)

    @staticmethod
    def _knn(x, y, k, radius, points):
        res = np.ones(shape=[k, 2])
        k = min(k, points.shape[0])
        if k > 0:
            nbrs = NearestNeighbors(
                n_neighbors=k, algorithm='ball_tree').fit(points)
            knn = nbrs.kneighbors(X=[[x, y]])
            knn_points = points[knn[1][knn[0] < radius]]

            res = knn_points - np.array([x, y])

            # res[:knn_points.shape[0], :] = np.concatenate([knn_points - np.array([x, y]), np.zeros(shape=[knn_points.shape[0], 1])], axis=1)


        return res


    def sense(self, x, y, k, radius):

        things_to_sense = [self.ant_db.points,
                           self.trace_db.points, self.food_db.points]

        # things_to_sense = [self.food_db.points]

        res = np.empty(shape=[0, INPUT_SIZE])

        for i, thing in enumerate(things_to_sense):
            sensed = self._knn(x, y, k, radius, thing)
            sensed = np.concatenate(
                [sensed, to_one_hot(i * np.ones(shape=[sensed.shape[0], 1], dtype=np.int32), len(things_to_sense))], axis=1)
            res = np.concatenate([res, sensed])
        
        return res

    def score(self):
        distances = distance.cdist(self.ant_db.points, self.food_db.points, 'euclidean')

        score = np.average(distances, axis=0)

        for ant in self.ants:
            ant.score = (ant.score + distances[ant.index]) / 2.0

        return score[0]


    def update(self):
        self.trace_db.update()
        

class Game(object):
    def __init__(self, receptive_area, max_food):
        # self.db = Database()
        self.world : World = World()
        self.receptive_area : int = receptive_area
        self.hidden : int = 256
        self.max_food : int= max_food
        self.score : float = 0
        self.steps : int = 0

    @staticmethod
    def rand_mlp(sizes):
        assert(len(sizes) >= 3)
        layers = []
        for input_size, output_size in sizes:
            # weights = np.random.random(size=[input_size, output_size]) - 0.5
            weights = np.random.normal(size=[input_size, output_size], scale=0.05)
            # bias = np.random.random(size=[output_size]) - 0.5
            bias = np.random.normal(size=[output_size], scale=0.05)
            layers.append((weights, bias))
        
        return layers


    def populate_world_randomly(self,num_ants, num_food):
        # ant_pos = rand_pos(FOOD_POS, ANT_RADIUS)
        ant_pos = ANT_POS
        # ant_pos = np.random.random(size=[2])
        layers = self.rand_mlp(
            [(INPUT_SIZE, self.hidden), 
             (self.hidden, self.hidden),
             (self.hidden, self.hidden),
             (self.hidden, self.hidden),
             (self.hidden, self.hidden), 
             (self.hidden, self.hidden), 
             (self.hidden * 2, OUTPUT_SIZE)])
        for i in range(num_ants):
            self.world.add_ant(Ant(index=-1, ant_width=ANT_SIZE, ant_height=ANT_SIZE, color=[
                0, 0, 0], world_width=width, world_height=height, layers=layers.copy(), receptive_area=self.receptive_area), ant_pos)

        for i in range(num_food):
            # food_pos = FOOD_POS
            food_pos = rand_pos(ANT_POS, ANT_RADIUS)
            self.world.add_food(Food(index=-1, amount=random.random() * self.max_food, food_width=FOOD_SIZE, food_height=FOOD_SIZE, color=[
                255, 0, 0], world_width=width, world_height=height), food_pos)

    def populate_world_from_other(self, other, k=1.0):
        # ant_pos = rand_pos(FOOD_POS, ANT_RADIUS)
        ant_pos = ANT_POS
        # ant_pos = np.random.random(size=[2])
        num_ants = len(other.world.ants)

        for i, ant in enumerate(itertools.cycle(sorted(other.world.ants, key=lambda x: x.get_score(), reverse=False)[:int(num_ants * k)])):
            if i == num_ants:
                break
            new_ant = Ant.fromother(ant)
            self.world.add_ant(new_ant, ant_pos)
        
        for i, food in enumerate(other.world.foods):
            # food_pos = FOOD_POS
            food_pos = rand_pos(ANT_POS, ANT_RADIUS)
            new_food = Food.fromother(food)
            self.world.add_food(new_food, food_pos)


    def copy(self):
        copied_game = Game(self.receptive_area, self.max_food)
        copied_game.populate_world_from_other(self)
        return copied_game

    def reset_score(self):
        for ant in self.world.ants:
            ant.score = 0

    def mutate(self):
        mutated_game = Game(self.receptive_area, self.max_food)
        mutated_game.populate_world_from_other(self, k=0.1)
        for ant in mutated_game.world.ants:
            ant.mutate()
        

        return mutated_game
        

    def play(self, steps, events, is_draw=True):
        dt = 0.1
        for step in range(steps):
            self.steps += 1


            self.world.ants.update(events, dt, self.world)
            self.world.foods.update(events, dt, self.world)

            self.score += self.world.score()

            self.world.ants.draw(screen)
            self.world.foods.draw(screen)

            self.world.update()

            if is_draw:
                for trace_v, trace_p in zip(self.world.trace_db.values, self.world.trace_db.points):
                    pygame.draw.circle(screen, [int(
                        trace_v * 255), 0, 0], (trace_p[0] * screen.get_width(), trace_p[1] * screen.get_height()), 3)


            # dt = clock.tick(60)

    def get_score(self):
        return self.score

        

def create_games(args):
    new_games = [Game(args.receptive_area, args.max_food) for x in range(args.num_of_games)]
    for game in new_games:
        game.populate_world_randomly(args.num_ants, args.num_food)
    return new_games

def mutate_games(games):
    new_games = [game.mutate() for game in games]
    for game in new_games:
        game.reset_score()
    return new_games


def copy_games(games):
    new_games = [game.copy() for game in games]
    for game in new_games:
        game.reset_score()
    return new_games


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Ants, again")
    parser.add_argument("--num_ants", type=int, default=10)
    parser.add_argument("--num_food", type=int, default=10)
    parser.add_argument("--max_food", type=int, default=1000)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--steps_per_epoch", type=int, default=50)
    parser.add_argument("--num_of_games", type=int, default=4)
    parser.add_argument("--receptive_area", type=int, default=5)

    args = parser.parse_args()

    pygame.init()

    size = width, height = args.width, args.height

    screen = pygame.display.set_mode(size)
    myfont = pygame.font.SysFont("mono", 16)

    score = 0
    generation = 0

    games = create_games(args)

    steps_per_epoch = args.steps_per_epoch

    is_draw = True
    
    while True:
        generation += 1
        for i, game in enumerate(games):
            for step in range(steps_per_epoch):
                events = pygame.event.get()
                for e in events:
                    if e.type == pygame.KEYDOWN:
                        if e.key == pygame.K_q:
                            exit()
                        if e.key == pygame.K_ESCAPE:
                            exit()

                        if e.key == pygame.K_LEFTBRACKET:
                            steps_per_epoch -= 1

                        if e.key == pygame.K_RIGHTBRACKET:
                            steps_per_epoch += 1

                        if e.key == pygame.K_SEMICOLON:
                            MUTATION_FACTOR *= 0.5

                        if e.key == pygame.K_QUOTE:
                            MUTATION_FACTOR *= 2.0

                        if e.key == pygame.K_d:
                            is_draw = not is_draw

                    if e.type == pygame.QUIT:
                        exit()

                if is_draw:
                    screen.fill((255, 255, 255))
                    scoretext = myfont.render(
                        "Game - {2} - G: {1} - S: {0:7.2f} (s/g: {3}) mf: {4}".format(game.get_score(), generation, i, steps_per_epoch, MUTATION_FACTOR), 1, (0, 0, 0))
                    screen.blit(scoretext, (5, 10))

                game.play(1, events, is_draw)

                if is_draw:
                    pygame.display.update()

        games = sorted(games, key=lambda x: x.get_score(), reverse=False)

        print("Generation {}: Best score: {}".format(
            generation, games[0].get_score()))
    
        num_mutated_games = args.num_of_games // 2
        num_best_games = args.num_of_games - num_mutated_games

        mutated_games = copy_games(games[:num_best_games]) + \
            mutate_games(games[:num_mutated_games])

        games = mutated_games



    
