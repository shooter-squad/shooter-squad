import copy
from typing import Tuple
import sys
import gym
import numpy as np
import torch

from individual import crossover, mutation, Individual, ranking_selection
from population import Population
from nn.base_nn import NeuralNetwork
from nn.rnn import RNN
from utils import plot_learning_curve, make_env

sys.path.insert(0, r'/home/elvinzhu/github/shooter-squad/haoyun_code')
from Env import *



class RNNIndividual(Individual):

    def get_model(self, input_size, hidden_size, output_size) -> NeuralNetwork:
        return RNN(input_size, hidden_size, 12, output_size)

    def run_single(self, env, n_episodes=2000, render=False) -> Tuple[float, np.array]:
        obs = env.reset().flatten()
        fitness = 0
        hidden = self.nn.init_hidden()
        for i in range(n_episodes):
            if render:
                env.render()
            obs = torch.from_numpy(obs).float()
            action, hidden = self.nn.forward(obs, hidden)
            action = action.detach().numpy()
            action = np.nan_to_num(action)

            obs, reward, done, _ = env.step(np.argmax(action))
            obs = obs.flatten()
            fitness += reward
            if done:
                break
        rc = self.nn.get_weights_biases()
        return fitness, rc


def generation(env, old_population, new_population, p_mutation, p_crossover, p_iversion=0.0):
    for i in range(0, len(old_population) - 1, 2):
        # Selection
        # parent1 = roulette_wheel_selection(old_population)
        # parent2 = roulette_wheel_selection(old_population)
        parent1, parent2 = ranking_selection(old_population)

        # Crossover
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)

        child1.weights_biases, child2.weights_biases = crossover(parent1.weights_biases,
                                                                 parent2.weights_biases,
                                                                 p_crossover)
        # Mutation
        child1.weights_biases = mutation(child1.weights_biases, p_mutation)
        child2.weights_biases = mutation(child2.weights_biases, p_mutation)

        # Update model weights and biases
        child1.update_model()
        child2.update_model()

        child1.calculate_fitness(env)
        child2.calculate_fitness(env)

        # If children fitness is greater thant parents update population
        if child1.fitness + child2.fitness > parent1.fitness + parent2.fitness:
            new_population[i] = child1
            new_population[i + 1] = child2
        else:
            new_population[i] = parent1
            new_population[i + 1] = parent2


if __name__ == '__main__':
    # env = gym.make('BipedalWalker-v3')
    # env.seed(123)

    env_name = 'shooter'
    env = make_env(env_name)

    #must be even
    POPULATION_SIZE = 20
    MAX_GENERATION = 500
    MUTATION_RATE = 0.2
    CROSSOVER_RATE = 0.8
    MUTATION_DECREMENT = 0.1/500

    INPUT_SIZE = env.observation_space.shape[0] * env.observation_space.shape[1] * env.observation_space.shape[2]
    print("input size:",INPUT_SIZE)
    HIDDEN_SIZE = 40
    OUTPUT_SIZE = 4

    p = Population(RNNIndividual(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE),
                   POPULATION_SIZE, MAX_GENERATION, MUTATION_RATE, CROSSOVER_RATE, 0.0, MUTATION_DECREMENT)
    p.run(env, generation, verbose=True, output_folder='outputs/output3/')

    env.close()