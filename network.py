# takes ins + outs, returns basic 1 layer network
# takes inputs, returns outputs
# creates neurons and connections

from neuron import Neuron
from connection import Connection
import numpy


class Network:
    consonants = (
        "b", "c", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p", "q", "r", "s", "t", "v", "w", "x", "y", "z")
    vowels = ("a", "e", "i", "o", "u")

    def __init__(self, ins, outs, recurrent, id, dna, basic, mutate, species, colour):
        self.ins = ins
        self.outs = outs
        self.layers = 1
        self.recurrent = recurrent
        self.neurons = []
        self.connections = []
        self.recurrents = []
        self.max_node = 0
        self.id = id
        self.fitness = 0
        self.adj_fitness = 0
        self.neuron_ids = []
        self.dna = dna
        self.mutation_amount = 0.1
        self.species = species
        if basic:
            self.setup_basic()
        else:
            if mutate:
                self.mutate_dna()
            self.translation_setup()
        self.name = self.generate_name()
        self.colour = colour

    def setup_basic(self):
        for i in range(self.ins):
            input_neuron = Neuron(0, self.max_node, True, 1)
            self.dna["neurons"][0].append(self.max_node)
            self.neuron_ids.append(self.max_node)
            self.next_neuron(input_neuron)

        for i in range(self.outs):
            self.dna["neurons"][1].append(self.max_node)
            self.neuron_ids.append(self.max_node)
            self.next_neuron(Neuron(1, self.max_node, True, 0))

        for i in range(self.ins):
            for j in range(self.outs):
                weight = numpy.random.uniform(-0.5, 0.5)
                connection = Connection(
                    self.neurons[i], self.neurons[j + self.ins], weight, True, False)
                self.connections.append(connection)
                self.neurons[i].add_connection(connection)
                self.dna["connections"].append([i, j + self.ins, weight, True])

    def translation_setup(self):
        layers = len(self.dna["neurons"])

        for i in range(self.ins):
            input_neuron = Neuron(0, i, True, 1)
            self.next_neuron(input_neuron)

        for i in range(layers - 2):
            for j in range(len(self.dna["neurons"][i + 1])):
                self.next_neuron(
                    Neuron(i + 1, self.dna["neurons"][i + 1][j], False, 0))

        for i in range(self.outs):
            self.next_neuron(Neuron(layers - 1, self.ins + i, True, 0))

        for i in range(len(self.dna["connections"])):
            if not self.dna["connections"][i][3]:
                continue
            code = self.dna["connections"][i]
            node1 = self.get_node(code[0])
            node2 = self.get_node(code[1])
            self.connections.append(Connection(
                node1, node2, code[2], code[3], False))
            node1.add_connection(self.connections[-1])

        for i in range(len(self.dna["recurrents"])):
            if not self.dna["recurrents"][i][3]:
                continue
            code = self.dna["recurrents"][i]
            node1 = self.get_node(code[0])
            node2 = self.get_node(code[1])
            self.recurrents.append(Connection(
                node1, node2, code[2], code[3], True))
            node1.recurrents.append(self.recurrents[-1])

    def feed_forward(self, inputs):
        do_print = False
        for i in range(len(self.neurons)):
            if self.neurons[i].layer == 0:
                self.neurons[i].set_value(inputs[i], do_print)

        for i in range(len(self.neurons)):
            self.neurons[i].fire_recurrents()
        outputs = []

        for i in range(self.max_node - self.outs, self.max_node):
            outputs.append(self.neurons[i].output_value)

        if do_print:
            print("\n", outputs)
            print("\n", self.dna)

        return outputs

    def next_neuron(self, neuron):
        self.neurons.append(neuron)
        self.max_node += 1

    def get_node(self, number):
        for n in self.neurons:
            if n.id == number:
                return n
        for i in self.neurons:
            print(i.id)
        print("can't find node" + str(number) + "!!")

    def reset(self):
        self.fitness = 0

    def mutate_dna(self):
        rand = numpy.random.rand()

        if rand < 0.001:
            self.mutate_nodes()
        elif rand < 0.0012:
            self.activate_or_inactivate()

        rand2 = numpy.random.rand()

        if rand2 < 0.04:
            self.mutate_connections()

        rand3 = numpy.random.rand()
        if rand3 < 0.8:
            self.mutate_weights()

    def mutate_weights(self):
        keyword = numpy.random.choice(("connections", "recurrents"))

        for c in range(len(self.dna[keyword])):
            rand = numpy.random.rand()

            if rand < 0.1:
                self.dna[keyword][c][2] = numpy.random.uniform(-1, 1)
            else:
                self.dna[keyword][c][2] += numpy.random.uniform(
                    -self.mutation_amount, self.mutation_amount)

            if self.dna[keyword][c][2] > 1:
                self.dna[keyword][c][2] = 1
            if self.dna[keyword][c][2] < -1:
                self.dna[keyword][c][2] = -1

    def mutate_nodes(self):
        c = numpy.random.randint(len(self.dna["connections"]))

        self.dna["connections"][c][3] = False

        start = self.dna["connections"][c][0]
        end = self.dna["connections"][c][1]
        weight = self.dna["connections"][c][2]
        start_layer = 0
        end_layer = 0

        max_node = 0
        for n in range(len(self.dna["neurons"])):
            this_max = max(self.dna["neurons"][n])
            max_node = this_max if this_max > max_node else max_node
            if start in self.dna["neurons"][n]:
                start_layer = n
            if end in self.dna["neurons"][n]:
                end_layer = n

        if start_layer + 1 == end_layer:
            self.dna["neurons"].insert(start_layer + 1, [max_node + 1])
        else:
            self.dna["neurons"][start_layer +
                                int((end_layer - start_layer) / 2)].append(max_node + 1)

        self.dna["connections"].append([start, max_node + 1, 1, True])
        self.dna["connections"].append([max_node + 1, end, weight, True])

    def activate_or_inactivate(self):
        keyword = numpy.random.choice(("connections", "recurrents"))

        length = len(self.dna[keyword])

        if length == 0:
            return

        c = numpy.random.randint(length)

        self.dna[keyword][c][3] = not self.dna[keyword][c][3]

    def mutate_connections(self):
        l1 = numpy.random.randint(len(self.dna["neurons"]))

        n1 = numpy.random.choice(self.dna["neurons"][l1])

        l2 = numpy.random.randint(len(self.dna["neurons"]))

        n2 = numpy.random.choice(self.dna["neurons"][l2])

        c_or_r = "connections" if l2 > l1 else "recurrents"

        for c in self.dna[c_or_r]:
            if c[0] == n1 and c[1] == n2:
                return

        self.dna[c_or_r].append([n1, n2, numpy.random.uniform(-1, 1), True])

    def generate_name(self):
        first_patt = numpy.random.choice((True, False))
        to_return = ""

        if first_patt:
            to_return += numpy.random.choice(self.consonants)
            to_return += numpy.random.choice(self.vowels)
            to_return += numpy.random.choice(self.consonants)
            to_return += numpy.random.choice(self.consonants)
            to_return += numpy.random.choice(self.vowels)
            to_return += numpy.random.choice(self.vowels)
        else:
            to_return += numpy.random.choice(self.vowels)
            to_return += numpy.random.choice(self.consonants)
            to_return += numpy.random.choice(self.consonants)
            to_return += numpy.random.choice(self.vowels)
            to_return += numpy.random.choice(self.consonants)
            to_return += numpy.random.choice(self.vowels)

        return to_return
