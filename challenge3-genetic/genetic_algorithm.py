#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# h/t: https://machinelearningmastery.com/simple-genetic-algorithm-from-scratch-in-python/

from numpy.random import rand
# genetic algorithm search of the one max optimization problem
from numpy.random import randint


# genetic algorithm
class GeneticAlgorithm():

    @classmethod
    def genetic_algorithm(cls, objective, n_bits, n_iter, n_pop, r_cross, r_mut):
        # initial population of random bitstring
        pop = [randint(0, 2, n_bits).tolist() for _ in range(n_pop)]
        # keep track of best solution
        best, best_eval = 0, objective(pop[0])
        # enumerate generations
        for gen in range(n_iter):
            # evaluate the fitness of all candidates in the population
            scores = [objective(c) for c in pop]
            # check for new best solution
            for i in range(n_pop):
                if scores[i] < best_eval:
                    best, best_eval = pop[i], scores[i]
                    print(">%d, new best f(%s) = %.3f" % (gen, pop[i], scores[i]))
            # select parents
            selected = [cls._selection(pop, scores) for _ in range(n_pop)]
            # create the next generation
            children = list()
            for i in range(0, n_pop, 2):
                # get selected parents in pairs
                p1, p2 = selected[i], selected[i + 1]
                # crossover and mutation
                for c in cls._crossover(p1, p2, r_cross):
                    # mutation
                    cls._mutation(c, r_mut)
                    # store for next generation
                    children.append(c)
            # replace population
            pop = children
        return [best, best_eval]

    @classmethod
    def _selection(cls, pop, scores, k=3):
        # first random selection
        selection_x = randint(len(pop))
        for x in randint(0, len(pop), k - 1):
            # check if better (e.g. perform a tournament)
            if scores[x] < scores[selection_x]:
                selection_x = x
        return pop[selection_x]

    @classmethod
    def _crossover(cls, p1, p2, r_cross):
        # children are copies of parents by default
        c1, c2 = p1.copy(), p2.copy()
        # check for recombination
        if rand() < r_cross:
            # select crossover point that is not on the end of the string
            pt = randint(1, len(p1) - 2)
            # perform crossover
            c1 = p1[:pt] + p2[pt:]
            c2 = p2[:pt] + p1[pt:]
        return [c1, c2]

    @classmethod
    def _mutation(cls, bitstring, r_mut):
        for i in range(len(bitstring)):
            # check for a mutation
            if rand() < r_mut:
                # flip the bit
                bitstring[i] = 1 - bitstring[i]
