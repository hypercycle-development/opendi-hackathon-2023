#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from genetic_algorithm import GeneticAlgorithm
from prompt_objective import PromptObjective

# define the total iterations
n_iter = 100
# bits
n_bits = 20
# define the population size
n_pop = 100
# crossover rate
r_cross = 0.9
# mutation rate
r_mut = 1.0 / float(n_bits)

target_output = 'simple, lively, strong'
pa = PromptObjective(n_bits, target_output)

# perform the genetic algorithm search
best_genotype, score = GeneticAlgorithm.genetic_algorithm(pa.objective,
                                                          n_bits,
                                                          n_iter,
                                                          n_pop,
                                                          r_cross,
                                                          r_mut)
best = pa.phenotype(best_genotype)
print('Done!')
print('f(%s) = %f' % (best, score))
