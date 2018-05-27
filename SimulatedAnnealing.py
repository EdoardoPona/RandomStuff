""" Implements simulated annealing to find out a string """
import random
import string
import math

all_chars = string.lowercase + string.digits + string.punctuation

rand_str = lambda n: ''.join([random.choice(all_chars) for i in xrange(n)])
objective = rand_str(500)
solution = rand_str(len(objective))


def cost(s):
    cost = 0
    for i, j in enumerate(s):
        if objective[i] != j:
            cost += 1
    return cost


def fitness(s):
    fitness = 0
    for i, j in enumerate(s):
        if objective[i] == j:
            fitness += 1
    return fitness


def generate_neighbor(s):
    index = random.randint(0, len(s)-1)
    # s = s.replace(s[index], random.choice(string.lowercase), 1)
    if len(s) == index -1:
        s = s[:index] + random.choice(all_chars)
    else:
        s = s[:index] + random.choice(all_chars) + s[index+1:]

    return s


"""def acceptance_prob(c_old, c_new, T):        # is using cost, not fitness
    return math.e * (c_old-c_new) / T"""


def acceptance_prob(f_new, f_old, T):
    return math.e * (f_new-f_old) / T


def anneal(solution):
    old_fitness = fitness(solution)
    T = 1.0
    T_min = 1e-5
    alpha = 0.99
    iters = 0
    while T > T_min:
        for i in xrange(250):
            new_solution = generate_neighbor(solution)
            new_fitness = fitness(new_solution)
            a_p = acceptance_prob(new_fitness, old_fitness, T)
            if a_p > random.random():
                solution, old_fitness = new_solution, new_fitness
            i += 1
        T *= alpha
        iters += 1
        if iters % 50 == 0:
            print cost(solution)
    return solution

solution = anneal(solution)
print cost(solution)
