import gym
import sys
import math
import matplotlib.pyplot as plt
import numpy as np
import time
import random
from colors import *

env = gym.make('Marvin-v0')
actions_len = 20                        # number of actions each Marvin has
population = 20                         # number of marvin in each generation
generations = 500                       # number of generations
train_steps = 4000                      # steps Marvin can make while training
walk_maxsteps = 10000                   # maximum steps Marvin can make while walking
selection_percent = 0.25                # top 25% Marvins survives each generation
mutation_chance = 0.1                   # chance to be changed for single action in a single marvin
highest_fitness = 1                     # current highest fitness
flags = {}                              # privided flags

class Marvin:

    def __init__(self, actions_len):
        self.actions = np.asarray([env.action_space.sample() for _ in range(actions_len)])
        self.fitness = 0


def genetic_algorithm(the_crowd):

    for generation in range(generations):

        print(color("\n Generation: ", bg='#555555'), color(" " + str(generation) + " ", bg='#555555'), "\n")

        the_crowd = fitness(the_crowd)

        if 'L' in flags:
            print_generation_log(the_crowd)

        the_crowd = selection(the_crowd)
        the_crowd = crossover(the_crowd)
        the_crowd = mutation(the_crowd)

        if any(marvin.fitness >= 300 for marvin in the_crowd):
            print ("\n ----- TRAINED! ----- \n")
            break

    return the_crowd


def print_generation_log(the_crowd):

    the_fitnesses = np.asarray(sorted([marvin.fitness for marvin in the_crowd], reverse=True)).T

    print ("Population fitnesses :\n", the_fitnesses, "\n")
    print ("The best actions    : ", the_crowd[0].actions[0], " ... ", the_crowd[0].actions[-1])
    print ("The best act hash   : ", sum(map(sum, the_crowd[0].actions)))
    print ("fitness average     : ", sum(the_fitnesses) / len(the_crowd))
    print ("fitness max         : ", max(the_fitnesses))
    print ("\n")


def init_crowd(population):

    return [Marvin(actions_len) for _ in range(population)]


def fitness(the_crowd):

    for marvin in the_crowd:
        observation = env.reset()
        for t in range(train_steps):

            observation, reward, done, info = env.step(marvin.actions[t % actions_len])
            marvin.fitness = marvin.fitness + reward

            if done:
                break

        check_actions(marvin)

    return the_crowd

def check_actions(marvin):
 
    global highest_fitness
    global flags

    if marvin.fitness > highest_fitness:

       highest_fitness = marvin.fitness

       if 's' in flags:

           print(color(" The best weights resaved ", bg='#117a40'), \
                   color(" New best fitness: ", bg='#117a40'), \
                   color(" " + str(highest_fitness) + " ", bg='#117a40'))
           np.save("actions", marvin.actions)


def selection(the_crowd):

    the_crowd = sorted(the_crowd, key=lambda marvin: marvin.fitness, reverse=True)
    the_crowd = the_crowd[:int(selection_percent * len(the_crowd))]

    for marvin in the_crowd:
        marvin.fitness = 0

    return the_crowd


def crossover(the_crowd):

    offspring = []

    for _ in range(int(((population - len(the_crowd)) / 2))):

        parent_1 = random.choice(the_crowd)
        parent_2 = random.choice(the_crowd)
        child_1 = Marvin(actions_len)
        child_2 = Marvin(actions_len)
        split = random.randint(0, actions_len)
        child_1.actions = np.concatenate((parent_1.actions[0:split], parent_2.actions[split:]))
        child_2.actions = np.concatenate((parent_2.actions[0:split], parent_1.actions[split:]))

        offspring.append(child_1)
        offspring.append(child_2)

    the_crowd.extend(offspring)

    return (the_crowd)


# The_crowd is always sorted by the fitness, the best Marvin always at index 0.
# A Marvin with the best fitness never mutate and remain all the weights. 
def mutation(the_crowd):

    for marvin in the_crowd[1:]:

        for idx, param in enumerate(marvin.actions):

            if random.uniform(0.0, 1.0) <= mutation_chance:

                marvin.actions[idx] = np.asarray(env.action_space.sample())

    return the_crowd


def walk(the_crowd):      

    global highest_fitness

    for marvin in the_crowd:

        observation = env.reset()

        for t in range(walk_maxsteps):

            env.render()
            observation, reward, done, info = env.step(marvin.actions[t % actions_len])
            marvin.fitness = marvin.fitness + reward

            if 'wl' in flags:
                print(color("\n Marvin: ", bg='#555555'), color(" " + str(the_crowd.index(marvin)) + " ", bg='#555555'), "\n",
                        #"observation : ", observation, "\n",
                        "reward      : ", reward, "\n", 
                        "fitness     : ", marvin.fitness, "\n")

            if done:
                break

    return the_crowd

#--------------------FLAGS-----------------------------------------------------

def display_help():

    print(color("\n Available commands: ", fg='#000000', bg='#bbbbbb'), "\n")

    print (color(" –-walk ", bg='#444444'), " or ", color(" -w ", bg='#444444'), \
            " Display only walking process.", "\n", sep='')
    print (color(" –-help ", bg='#444444'), " or ", color(" -h ", bg='#444444'), \
            " Display available commands.", "\n", sep='')
    print (color(" –-load ", bg='#444444'), " or ", color(" -l ", bg='#444444'), \
            " File Load weights for Marvin agent from a file. Skip training process if this option is specified.", "\n", sep='')
    print (color(" –-logs ", bg='#444444'), " or ", color(" -L ", bg='#444444'), \
            " Display training logs.", "\n", sep='')
    print (color(" –-save ", bg='#444444'), " or ", color(" -s ", bg='#444444'), \
            " File Save weights to a file after running the program.", "\n", sep='')
    print (color(" –-walking-logs ", bg='#444444'), " or ", color(" -wl ", bg='#444444'), \
            " Display logs while walking.", "\n",  sep='')

    print (color("\n Example: ", fg='#000000', bg='#bbbbbb'), "\n")

    print (color(" python3 Marvin.py -L -l actions.npy -s ", bg='#444444'), \
            " Will load weights from file, continue training based on that weights and save new best performing weights", "\n", sep='')


def parse_flags(flags):

    args = sys.argv[1:]
    f = 0;

    for arg in args:

        # HELP
        if arg == '-h' or arg == '--help':

            if len(sys.argv) != 2:

                print (color("\n Error: Too many arguments ", bg='#7a1124'), "\n")
                sys.exit(1)

            display_help()
            sys.exit(1)

        # WALK
        elif arg == '-w' or arg == '--walk':

            key = 'w'

            if key in flags:

                print (color("\n Error: Flag " + str(arg) + " was lready provided ", bg='#7a1124'), "\n")
                display_help()
                sys.exit(1)
            
            flags[key] = ''

        # LOAD
        elif arg == '-l' or arg == '--load':

            key = 'l'

            if key in flags:

                print (color("\n Error: Flag " + str(arg) + " was lready provided ", bg='#7a1124'), "\n")
                display_help()
                sys.exit(1)

            try:
            
                flags[key] = sys.argv[sys.argv.index(arg) + 1]
                f = 1;

            except:

                print (color("\n Error: No file provided ", bg='#7a1124'), "\n")
                display_help()
                sys.exit(1)

        # SAVE
        elif arg == '-s' or arg == '--save':
        
            if arg in flags:

                print (color("\n Error: Flag " + str(arg) + " was lready provided ", bg='#7a1124'), "\n")
                display_help()
                sys.exit(1)

            flags['s'] = ''

        # TRAINING LOGS
        elif arg == '-L' or arg == '--logs':

            key = 'L'

            if key in flags:

                print (color("\n Error: Flag " + str(arg) + " was lready provided ", bg='#7a1124'), "\n")
                display_help()
                sys.exit(1)
            
            flags[key] = ''

        # WALKING LOGS
        elif arg == '-wl' or arg == '--walking-logs':

            key = 'wl'

            if key in flags:

                print (color("\n Error: Flag " + str(arg) + " was lready provided ", bg='#7a1124'), "\n")
                display_help()
                sys.exit(1)
            
            flags[key] = ''


        # NOT A FLAG
        else:

            if f:

                f = 0
                continue
            
            print (color("\n Error: Flag does not exist ", bg='#7a1124'), "\n")
            display_help();
            sys.exit(0);


if __name__ == '__main__':

    the_crowd = init_crowd(population)
    parse_flags(flags)

    if 'l' in flags:

        try: 
                actions = np.load(flags['l'])

                for marvin in the_crowd:
                    marvin.actions = actions
               
        except:
            
            print (color("\n Eroor: File does not exist ", bg='#7a1124'), "\n")
            sys.exit(2)

    if 'w' in flags:
        
        walk(the_crowd)

    else:
    
        the_crowd = genetic_algorithm(the_crowd)
        walk(the_crowd)
