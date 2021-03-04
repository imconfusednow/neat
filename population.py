# takes in pop size, returns networks
# takes in ids => fitness dict. Returns new population
# creates species and networks
from network import Network
from species import Species
import math
import numpy

class Population:
    def __init__(self, size, ins, outs, recurrent):
        self.size = size
        self.networks = []
        self.max_id = 0
        self.max_species_id = 1
        self.generation = 0
        self.cull = int(0.5 * self.size)
        for i in range(size):
            self.networks.append(
                Network(ins, outs, recurrent, self.max_id, {"neurons": [[], []], "connections": [], "recurrents": []}, True, False, 0, (255, 0, 0)))
            self.max_id += 1
        self.species = [Species(0, self.networks[0].dna, (255, 0, 0))]
        # self.neuron_ids = set()
        self.this_to_add = 0
        self.fitness_sum = 0
        self.chosen = dict()
        self.divergence_threshold = 1.5

    def get_pop(self):
        return self.networks

    def next_gen(self, pop):
        self.chosen.clear()
        self.generation += 1
        string = "////////////" + str(self.generation) + "///////////////"
        if len(self.species) > 1:
            string += " " + str(len(self.species[1].members))
        print(string)
        self.speciate(pop)
        self.extinct_species(0)
        self.cull_and_replace()
        self.extinct_species(0)
        self.networks = []
        for s in self.species:
            for i in s.chosen:
                self.chosen[str(i)] = s.id
            self.networks += s.members
        for p in self.networks:
            p.reset()

        # self.get_unique_ids()

        return self.networks

    def cull_and_replace(self):
        fitness_sum = 0
        pass_on_extra = 0

        starting_id = self.size * self.generation

        for s in self.species:
            fitness = s.set_fitness()
            fitness_sum += fitness

        self.fitness_sum = fitness_sum

        pop_track = 0

        ratio = 0
        ratios = []

        self.species = sorted(self.species, key=lambda x: x.fitness, reverse=True)

        for i in range(len(self.species) - 1, -1, -1):
            s = self.species[i]
            ratio = 0.25 / (1.25**len(self.species) - 1) if ratio == 0 else ratio * 1.25
            ratios.append(ratio)
            to_add = math.floor(self.size * ratio) + pass_on_extra

            if len(s.members) > 5:
                pass_on_extra = s.birth(to_add, True, starting_id)
            else:
                pass_on_extra = s.birth(to_add, False, starting_id)
            starting_id += to_add
            pop_track += to_add

        if pop_track < self.size:
            to_add = self.size - pop_track
            self.species[0].birth_extra(to_add, starting_id)
            starting_id += to_add

    def speciate(self, pop):
        for s in self.species:
            s.members = []
        self.species = sorted(self.species, key=lambda x: x.fitness, reverse=True)
        for p in range(len(pop) - 1, -1, -1):
            closest = 9999
            species = None
            for i in range(len(self.species)):
                s = self.species[i] 
                check_stale = False if i <= 5 else True
                how_close = s.matches_species(pop[p], check_stale)
                if how_close < closest:
                    species = s
                    closest = how_close
            if closest > self.divergence_threshold:
                self.species.append(Species(self.max_species_id, pop[p].dna, (numpy.random.randint(255),numpy.random.randint(255),numpy.random.randint(255))))
                self.max_species_id += 1
                self.species[-1].members.append(pop[p])
            else:
                species.members.append(pop[p])

    def extinct_species(self, min):
        for i in range(len(self.species) - 1, -1, -1):
            if len(self.species[i].members) <= min:
                print("Species " + str(self.species[i].id) + "went extinct :/")
                del self.species[i]

    def get_unique_ids(self):
        self.neuron_ids.clear()
        for n in self.networks:
            self.neuron_ids.update(n.neuron_ids)
