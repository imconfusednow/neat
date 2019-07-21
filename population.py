# takes in pop size, returns networks
# takes in ids => fitness dict. Returns new population
# creates species and networks
from network import Network
from species import Species
import math


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
				Network(ins, outs, recurrent, self.max_id, {"neurons": [[], []], "connections": []}, True, False))
			self.max_id += 1
		self.species = [Species(0, self.networks[0].dna)]
		self.neuron_ids = set()
		self.this_to_add = 0
		self.fitness_sum = 0

	def get_pop(self):
		return self.networks

	def next_gen(self, pop):
		self.generation += 1
		string = "////////////" + str(self.generation) + "///////////////" + str(
			len(self.species[0].members)) + " " + str(self.species[0].id)
		if len(self.species) > 1:
			string += " " + str(len(self.species[1].members))
		print(string)
		print(len(self.species))
		self.speciate(pop)
		self.extinct_species()
		self.cull_and_replace()
		self.extinct_species()
		self.networks = []
		for s in self.species:
			self.networks += s.members
		for p in self.networks:
			p.reset()

		self.get_unique_ids()

		return self.networks

	def cull_and_replace(self):
		fitness_sum = 0
		pass_on_extra = 0

		starting_id = self.size * self.generation

		for s in self.species:
			fitness = s.set_fitness()
			fitness_sum += fitness

		self.fitness_sum = fitness_sum

		for s in self.species:
			to_add = math.floor(s.fitness / fitness_sum * self.size) + pass_on_extra
			if s.id == 0: self.this_to_add = to_add
			# print("ta", s.fitness, fitness_sum, self.size)
			# print(s.template_genome)
			if len(s.members) > 5:
				pass_on_extra = s.birth(to_add, True, starting_id)
			else:
				pass_on_extra = s.birth(to_add, False, starting_id)
			starting_id += to_add

	def speciate(self, pop):
		for s in self.species:
			s.members = []
		for p in range(len(pop) - 1, -1, -1):
			matched = False
			for s in self.species:
				matched = s.matches_species(pop[p])
				if matched == True:
					break
			if not matched:
				self.species.append(Species(self.max_species_id, pop[p].dna))
				self.max_species_id += 1
				self.species[-1].members.append(pop[p])

	def extinct_species(self):
		for i in range(len(self.species) - 1, -1, -1):
			if len(self.species[i].members) == 0:
				print("Species " + str(self.species[i].id) + " Extinct :/")
				del self.species[i]

	def get_unique_ids(self):
		self.neuron_ids.clear()
		for n in self.networks:
			self.neuron_ids.update(n.neuron_ids)



	def save(self,nets):
	    save_file = open('H:\\Documents\\top_player-' + str(pid) + '-.pickle', mode='wb')
	    pickle.dump(player.brain, save_file)
	    print("player saved to ", save_file)
	    save_file.close()

def load(name, latest):
    path = 'H:\\Documents\\'
    if not (this_id):
        this_id = get_top_id()
    pickle_in = open("H:\\Documents\\top_player-" + str(this_id) + "-.pickle", "rb")
    return(pickle.load(pickle_in))