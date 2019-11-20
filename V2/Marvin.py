import gym
import sys
import math
import matplotlib.pyplot as plt
import numpy as np
import time
import random

flags = {}
env = gym.make('Marvin-v0')
actions_len = 20
population = 20
generations = 3500
steps = 3000                     # steps to render;
selection_percent = 0.25         # top 20% performance;
mutation_chance = 0.1            # chance for single action in a single marvin
highest_fitness = 1

class Marvin:

    def __init__(self, actions_len):
        self.actions = [env.action_space.sample() for _ in range(actions_len)]
        self.fitness = 0

   # def __str__(self):
    #    print (self.actions)


def genetic_algorithm(the_crowd):

    for generation in range(generations):

        print("Generation: ", generation)

        the_crowd = fitness(the_crowd)
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


def fitness(the_crowd, render=False): #add: logs=True or check the global dict for what to do. 

    global highest_fitness

    for marvin in the_crowd:
        observation = env.reset()
        for t in range(steps):

            if render:
                env.render()

            observation, reward, done, info = env.step(marvin.actions[t % actions_len])
            marvin.fitness = marvin.fitness + reward

            if done:
                break
    highest_fitness = check_actions(marvin, highest_fitness)

    return the_crowd

def check_actions(marvin, highest_fitness):

    if marvin.fitness > highest_fitness:
       highest_fitness = marvin.fitness
       with open("./actions.marvin", "w") as f:
           print(marvin.actions, marvin.fitness, file=f)

    return highest_fitness


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
        child_1.actions = parent_1.actions[0:split]
        child_1.actions.extend(parent_2.actions[split:])
        child_2.actions = parent_1.actions[0:split]
        child_2.actions.extend(parent_2.actions[split:])

        offspring.append(child_1)
        offspring.append(child_2)

    the_crowd.extend(offspring)

    return (the_crowd)


def mutation(the_crowd):

    #the_crowd = sorted(the_crowd, key=lambda marvin: marvin.fitness, reverse=True)

    for marvin in the_crowd[1:]:

        for idx, param in enumerate(marvin.actions):

            if random.uniform(0.0, 1.0) <= mutation_chance:

                marvin.actions[idx] = env.action_space.sample()

    return the_crowd

#--------------------FLAGS-----------------------------------------------------

def analize_flags(flags):

    flags += sys.argv[1]

if __name__ == '__main__':

   # flags = analize_flags(flags)
    the_crowd = init_crowd(population)
    the_crowd = genetic_algorithm(the_crowd)
    steps = 10000
    fitness(the_crowd, render=True)

   # steps = 60000
   # actions = [ [-0.44316366,  0.4826049 ,  0.74409175,  0.3973649 ],    [ 0.62588626, -0.03820594, -0.7448107 , -0.5599956 ],    [0.8787345 , 0.46926227, 0.33920595, 0.96323174],    [ 0.32581845, -0.12342902, -0.9709807 ,  0.7974424 ],    [-0.06966726, -0.70557904, -0.90222424, -0.14505629],    [0.15690647, 0.24020338, 0.56450385, 0.2284611 ],    [ 0.76510537,  0.31225076, -0.02183986,  0.86548626],    [ 0.9267913 ,  0.33261538, -0.58610106,  0.56980205],    [ 0.16678098, -0.75123906, -0.12122045, -0.40721872],    [-0.62262666, -0.78702337, -0.5271521 , -0.22636952],    [-0.5259644 , -0.21002115, -0.19919795, -0.68403184],    [-0.19033256, -0.83894277, -0.7061103 ,  0.04366134],    [-0.7154191 ,  0.00980125,  0.97144425, -0.93974704],    [-0.5962911 , -0.78028774, -0.67941487,  0.8744608 ],    [ 0.9251486 , -0.5703789 , -0.11347997,  0.02906036],    [-0.66267467,  0.62741256, -0.14445227,  0.1047221 ],    [-0.95188206, -0.195166  ,  0.5116838 , -0.35435733],    [ 0.34621856, -0.93653   ,  0.08295925, -0.727807  ],    [ 0.7884405 ,  0.7772044 ,  0.44183257, -0.29724872],    [ 0.24467027, -0.98358685,  0.7867034 , -0.71225536]]
   # actions = [ [ 0.14205632, -0.08324117,  0.02290588, -0.9923353 ],    [ 0.21472216, -0.00595039, -0.71475726, -0.47178075],    [-0.75005144,  0.7108593 ,  0.7889165 ,  0.04984283],    [ 0.00808224,  0.02944147, -0.45654348,  0.19148672],    [-0.6942348 ,  0.05026141,  0.60411084,  0.26911557],    [-0.52670896,  0.2886911 ,  0.36876222,  0.6212376 ],    [-0.556873  ,  0.9487755 ,  0.71650285,  0.6184399 ],    [ 0.48523852, -0.42995209,  0.76148325, -0.8630479 ],    [-0.67485756, -0.7618321 ,  0.35493946, -0.49862328],    [ 0.04430054, -0.45279568, -0.71466863, -0.30711636],    [ 0.9720895 , -0.49235353, -0.5535092 , -0.70820165],    [-0.72286594, -0.07382893,  0.17893128,  0.03385938],    [ 0.96015334,  0.21854188,  0.41359422, -0.99120003],    [-0.89037955, -0.7718009 , -0.02454801, -0.8063121 ],    [-0.1828346 ,  0.7344795 ,  0.42155516, -0.9170188 ],    [-0.520065  ,  0.95024425,  0.75766486,  0.91652805],    [-0.62035435,  0.52087116,  0.7187454 ,  0.76063645],    [0.94003654, 0.685116  , 0.9220782 , 0.7654703 ],    [-0.1009342 ,  0.52999264,  0.7271779 ,  0.517994  ],    [-0.80912846,  0.00571404, -0.01663941, -0.40815353]]
   # actions = [ [ 0.09120423, -0.7565104 , -0.48297235, -0.864336  ],    [-0.7078773, -0.6421431, -0.5125494, -0.2541883],    [-0.5528691 , -0.8745956 , -0.46035886, -0.14404973],    [ 0.34049204, -0.7451652 , -0.39855617, -0.4240937 ],    [-0.89219797, -0.8441836 , -0.724855  , -0.46736956],    [-0.62603956, -0.8700205 , -0.12108438, -0.1794597 ],    [ 0.5534296 ,  0.07798058,  0.68610996, -0.44921032],    [-0.03377989, -0.35462227, -0.696625  , -0.5517279 ],    [ 0.99512   , -0.10504565, -0.19945009,  0.11618579],    [-0.2551846 , -0.7307174 ,  0.56183094,  0.6597681 ],    [ 0.72491026,  0.8130031 , -0.05613614, -0.8318588 ],    [-0.2914871 , -0.16405745,  0.3664326 ,  0.27282307],    [ 0.14338514,  0.702083  , -0.29400223,  0.6401197 ],    [ 0.9664224 ,  0.4955296 , -0.03279333, -0.02644475],    [ 0.58008987,  0.17604512, -0.01203384,  0.9523065 ],    [-0.3011469 ,  0.16589928, -0.89775175,  0.74188256],    [0.07936992, 0.80687517, 0.27413833, 0.90870535],    [ 0.6534422 ,  0.04932126, -0.0402469 ,  0.8830367 ],    [0.74921906, 0.39137784, 0.5245959 , 0.09599676],    [-0.28884867, -0.8338717 , -0.44735065, -0.16961405]]

  #  the_crowd = init_crowd(5)
  #  for marvin in the_crowd:
  #      marvin.actions = actions
  #  fitness(the_crowd, render=True)
