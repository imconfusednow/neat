# takes matching networks

from network import Network

import copy
import math
import numpy


class Species:

    def __init__(self, id, genome, colour):
        self.id = id
        self.members = []
        self.template_genome = genome
        self.gene_coef = 1.0
        self.weight_coef = 0.4
        self.fitness = 0
        self.last_fitness = 0
        self.stale = False
        self.age = 0
        self.add_up = 0
        self.count = 0
        self.colour = colour
        self.chosen = []

    def birth(self, to_add, champ, starting_id):
        new_pop = []
        self.chosen = []
        self.add_up = 0
        self.count = 0
        self.sort_species(self.members)
        if len(self.members) < 1:
            self.members = []
            return to_add
        if champ:
            new_pop.append(Network(self.members[0].ins, self.members[0].outs, self.members[0].recurrent, self.members[0].id,
                                   copy.deepcopy(self.members[0].dna), False, False, self.id, self.colour))
            to_add -= 1

        to_cross_over = math.ceil(to_add * 0.2)
        to_add -= to_cross_over

        for i in range(to_cross_over):
            mem1 = self.choose_net()
            mem2 = self.choose_net()
            new_pop.append(Network(mem1.ins, mem1.outs, mem1.recurrent, starting_id + i,
                                   self.crossover(copy.deepcopy(mem1), copy.deepcopy(mem2)), False, True, self.id, self.colour))

        for i in range(to_add):
            mem = self.choose_net()
            new_pop.append(Network(mem.ins, mem.outs, mem.recurrent, starting_id +
                                   to_cross_over + i, copy.deepcopy(mem.dna), False, True, self.id, self.colour))

        self.members = new_pop

        self.age += 1

        return 0

    def birth_extra(self, to_add, starting_id):
        for i in range(to_add):
            mem = numpy.random.choice(self.members)
            self.members.append(Network(mem.ins, mem.outs, mem.recurrent,
                                        starting_id + i, copy.deepcopy(mem.dna), False, True, self.id, self.colour))

    def cull(self, pop):
        if self.age == 0:
            return
        self.members = pop[0: math.floor(len(self.members) * 0.5)]
        self.set_fitness()

    def choose_net(self):
        tot = 0
        rand = numpy.random.uniform(0, self.fitness * 0.5)
        for m in range(len(self.members)):
            i = self.members[m]
            tot += i.adj_fitness
            if tot >= rand:
                self.add_up += m
                self.count += 1
                self.chosen.append(i.id)
                return i

        tot = 0
        for i in self.members:
            tot += i.fitness

        # print(tot)
        # print(rand, self.fitness)
        print(self.id)

        print("No network choosen. Brace for crash!!")
        print(self.members)

    def matches_species(self, potential, check_stale):
        if self.stale:
            return 9999
        if self.age != 0 and self.age % 40 == 0:
            if check_stale and self.fitness < self.last_fitness:
                self.stale = True
                print("species ", self.id, " has gone stale. Age: ", self.age,
                      " last_fitness ", self.last_fitness, " fitness ", self.fitness)
                return 9999
            else:
                self.stale = False
                self.last_fitness = self.fitness
        return self.compare_genome(potential)


    def compare_genome(self, potential):
        genome = potential.dna
        divergence = 0
        different_genes = 0
        weight_difference = 0
        matches = 0

        if len(genome["connections"]) > len(self.template_genome["connections"]):
            gen1 = genome["connections"]
            gen2 = self.template_genome["connections"]
        else:
            gen1 = self.template_genome["connections"]
            gen2 = genome["connections"]

        for c in gen1:
            for c2 in gen2:
                matched = False
                if c[0] == c2[0] and c[1] == c2[1]:
                    weight_difference += abs(c[2] - c2[2])
                    matches += 1
                    matched = True
                    break
            if not matched:
                different_genes += 1

        if matches == 0:
            return False

        weight_difference /= matches

        genome_normaliser = len(genome["connections"]) - 60.0
        if genome_normaliser < 1:
            genome_normaliser = 1

        divergence = ((self.gene_coef * different_genes) /
                      genome_normaliser) + (self.weight_coef * weight_difference)

        return divergence

    def sort_species(self, curr_pop):
        self.members = sorted(
            curr_pop, key=lambda x: x.adj_fitness, reverse=True)

    def set_fitness(self):
        self.fitness = 0
        div = len(self.members)
        for m in self.members:
            m.adj_fitness = m.fitness / div
            self.fitness += m.adj_fitness

        return self.fitness

    def crossover(self, mem1, mem2):
        if mem1.id == mem2.id:
            return mem1.dna

        dna1 = mem1.dna if mem1.fitness > mem2.fitness else mem2.dna
        dna2 = mem2.dna if mem1.fitness > mem2.fitness else mem1.dna

        daughter_dna = dna1

        for i in range(len(daughter_dna["connections"])):
            for j in range(len(dna2["connections"])):
                if (daughter_dna["connections"][i][0] == dna2["connections"][j][0]) and (daughter_dna["connections"][i][1] == dna2["connections"][j][1]):
                    rand = numpy.random.rand()
                    if rand < 0.5:
                        daughter_dna["connections"][i][2] = dna2["connections"][j][2]

        return daughter_dna
