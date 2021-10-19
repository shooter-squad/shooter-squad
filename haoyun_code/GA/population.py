import copy
from datetime import datetime
from typing import Callable

import numpy as np
import torch
import time
import matplotlib.pyplot as plt

from individual import statistics
from timing import timing



class Population:
    def __init__(self, individual, pop_size, max_generation, p_mutation, p_crossover, p_inversion, p_mutation_decrease):
        self.pop_size = pop_size
        self.max_generation = max_generation
        self.p_mutation = p_mutation
        self.p_mutation_decrease = p_mutation_decrease
        self.p_crossover = p_crossover
        self.p_inversion = p_inversion
        self.old_population = [copy.copy(individual) for _ in range(pop_size)]
        self.new_population = []

    def set_population(self, population: list):
        self.old_population = population

    @timing
    def run(self, env, run_generation: Callable, verbose=False, log=False, output_folder=None, save_as_pytorch=False):
        best_model = sorted(self.old_population, key=lambda ind: ind.fitness, reverse=True)[0]
        t = time.time()
        mean_l = []
        for i in range(self.max_generation):

            for p in self.old_population:  
                p.calculate_fitness(env)
                

            print("Time: ", time.time()-t, np.mean( self.old_population[-1].weights_biases))
            t = time.time()

               
            self.new_population = [None for _ in range(self.pop_size)]
            run_generation(env,
                           self.old_population,
                           self.new_population,
                           self.p_mutation,
                           self.p_crossover,
                           self.p_inversion)

            if log:
                self.save_logs(i, output_folder)

            if verbose:
                a,_,_ = self.show_stats(i)
                mean_l.append(a)
                

            if (i%10 == 9):
                plt.figure(0)
                plt.scatter(np.arange(i+1),mean_l)
                np.save('plot/overall3/Gen'+str(i), np.array(mean_l))
                plt.savefig('plot/overall3/Gen'+str(i)+'.jpg')
                plt.close()
                plt.figure(1)
                plt.scatter(np.arange(self.pop_size),np.array([k.fitness for k in self.new_population]))
                plt.savefig('plot/genSample3/Gen'+str(i)+'.jpg')
                plt.close()
                print("Plot saved......\n")
                

            self.update_old_population()

            new_best_model = self.get_best_model_parameters()

            if new_best_model.fitness > best_model.fitness:
                print('Saving new best model with fitness: {}'.format(new_best_model.fitness))
                self.save_model_parameters(output_folder, i, save_as_pytorch)
                best_model = new_best_model

            self.p_mutation -= self.p_mutation_decrease
            print(self.p_mutation)

        if output_folder:
            self.save_model_parameters(output_folder, self.max_generation, save_as_pytorch)

        

    def save_logs(self, n_gen, output_folder):
        """
        CSV format -> date,n_generation,mean,min,max
        """
        date = self.now()
        file_name = 'logs.csv'
        mean, min, max = statistics(self.new_population)
        stats = f'{date},{n_gen},{mean},{min},{max}\n'
        with open(output_folder + file_name, 'a') as f:
            f.write(stats)

    def show_stats(self, n_gen):
        mean, min, max = statistics(self.new_population)
        date = self.now()
        stats = f"{date} - generation {n_gen + 1} | mean: {mean}\tmin: {min}\tmax: {max}\n"
        print(stats)
        return mean, min, max

    def update_old_population(self):
        self.old_population = copy.deepcopy(self.new_population)

    def save_model_parameters(self, output_folder, iterations, save_as_pytorch=False):
        best_model = self.get_best_model_parameters()
        file_name = self.get_file_name(self.now()) + f'_I={iterations}_SCORE={best_model.fitness}.npy'
        output_filename = output_folder + '-' + file_name
        if save_as_pytorch:
            torch.save(best_model.weights_biases, output_filename)
        else:
            np.save(output_filename, best_model.weights_biases)

    def get_best_model_parameters(self) -> np.array:
        """
        :return: Weights and biases of the best individual
        """
        return sorted(self.new_population, key=lambda ind: ind.fitness, reverse=True)[0]

    def get_file_name(self, date):
        return '{}_NN={}_POPSIZE={}_GEN={}_PMUTATION_{}_PCROSSOVER_{}'.format(date,
                                                                              self.new_population[0].__class__.__name__,
                                                                              self.pop_size,
                                                                              self.max_generation,
                                                                              self.p_mutation,
                                                                              self.p_crossover)

    @staticmethod
    def now():
        return datetime.now().strftime('%m-%d-%Y_%H-%M')