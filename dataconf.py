import pygame
import os
import json
from network import Network
import copy


class DataConf:

    def __init__(self, win):
        self.win = win
        self.font = pygame.font.SysFont("comicsansms", 12)

    def draw_brain(self, nets):
        state = 1
        net_num = 0

        while state < 3:

            events = pygame.event.get()
            for event in events:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        state =  1 if state == 0 else 0
                    if event.key == pygame.K_e:
                        state = 2
                    if event.key == pygame.K_ESCAPE:
                        state = 3
                    if event.key == pygame.K_UP:
                        net_num = net_num + 1 if net_num < len(nets) - 1 else 0
                    if event.key == pygame.K_DOWN:
                        net_num = net_num - 1 if net_num > 0 else len(nets) - 1
                    if event.key == pygame.K_f:
                        nets[net_num].feed_forward([1,1,1,1,1,1])
                if event.type == pygame.QUIT:
                    exit()

            if state == 0:
                self.draw_dna(nets[net_num].dna)
            elif state == 1:
                self.draw_net(nets[net_num])
            elif state == 2:
                nets[net_num] = self.edit_net(nets[net_num])
                state = 0
            else:
                return

            string = "ID: " + str(nets[net_num].id) + "        Species: " + str(
                nets[net_num].species) + "         Fitness: " + str(nets[net_num].fitness)
            self.text(string, 50, 10)

            pygame.display.update()

    def draw_dna(self, dna):
        locations = {}
        self.win.fill((255, 255, 255))
        width, height = pygame.display.get_surface().get_size()

        layers = len(dna["neurons"])
        for n in range(len(dna["neurons"])):
            num = len(dna["neurons"][n])
            offsetw = width / layers
            offseth = height / num
            for m in range(len(dna["neurons"][n])):
                x = n * offsetw + offsetw / 2 - 25
                y = (m * offseth + offseth / 2) - 25
                pygame.draw.rect(self.win, (255, 0, 0), (x, y, 50, 50))
                locations[dna["neurons"][n][m]] = (x + 25, y + 25)

        for i in range(len(dna["connections"])):
            connection = dna["connections"][i]
            colour = (255, 0, 0) if connection[2] < 0 else (0, 255, 255)
            if not connection[3]:
                colour = (200, 200, 200)
            pygame.draw.line(self.win, colour, (locations[connection[0]][0], locations[connection[0]][1]), (
                locations[connection[1]][0], locations[connection[1]][1]), 2)

        for i in range(len(dna["recurrents"])):
            connection = dna["recurrents"][i]
            colour = (255, 255, 0)
            if not connection[3]:
                colour = (200, 200, 200)
            pygame.draw.line(self.win, colour, (locations[connection[0]][0], locations[connection[0]][1]), (
                locations[connection[1]][0], locations[connection[1]][1]), 2)

    def draw_net(self, net):
        locations = {}
        self.win.fill((255, 255, 255))
        width, height = pygame.display.get_surface().get_size()

        layers = -1

        neurons = []

        for i in net.neurons:
            if i.layer > layers:
                layers = i.layer
                neurons.append([i])
            else:
                neurons[i.layer].append(i)

        layers += 1

        for n in range(len(neurons)):
            num = len(neurons[n])
            offsetw = width / layers
            offseth = height / num
            for m in range(len(neurons[n])):
                x = n * offsetw + offsetw / 2 - 25
                y = (m * offseth + offseth / 2) - 25
                pygame.draw.rect(self.win, (255, 0, 0), (x, y, 50, 50))
                locations[neurons[n][m].id] = (x + 25, y + 25)
                self.text(str(neurons[n][m].output_value), x + 22, y + 25)

        for i in range(len(net.connections)):
            connection = net.connections[i]
            colour = (255, 0, 0) if connection.weight < 0 else (0, 255, 255)
            if not connection.enabled:
                colour = (200, 200, 200)
            pygame.draw.line(self.win, colour, (locations[connection.from_node.id][0], locations[connection.from_node.id][1]), (
                locations[connection.to_node.id][0], locations[connection.to_node.id][1]), 2)

        for i in range(len(net.recurrents)):
            connection = net.recurrents[i]
            colour = (255, 255, 0)
            if not connection.enabled:
                colour = (200, 200, 200)
            pygame.draw.line(self.win, colour, (locations[connection.from_node.id][0], locations[connection.from_node.id][1]), (
                locations[connection.to_node.id][0], locations[connection.to_node.id][1]), 2)

    def edit_net(self, net):

        dna = {'neurons': [[0, 1, 2, 3, 4, 5], [9, 10], [6, 7, 8]], 'connections': [[0, 6, -0.2061805015151711, True], [0, 7, -0.81598605955629565, True], [0, 8, 1.9456378022000843, False], [1, 6, 0.0905799405955569, True], [1, 7, -0.7475655532645034, True], [1, 8, 0.0166163239207429, True], [2, 6, 1.7117701153467383, True], [2, 7, 0.1140451096041344, True], [2, 8, 1.0428636869490303, True], [3, 6, 1.477374080070203, True], [3, 7, -0.3306673681441799, True], [3, 8, 1.1335755676461518, True], [4, 6, 1.7553591149367705, True], [4, 7, -0.2238621083580107, True], [4, 8, 1.4527848669343368, True], [5, 6, 0.5781573776903348, False], [5, 7, -0.1857069801620698, True], [5, 8, 0.2479521709786687, True], [5, 9, 1.4323764583197623, True], [9, 6, 2, True], [0, 10, -1.188096159659686, True], [10, 8, 0.0192695727082604, True]], 'recurrents': []}

        net = Network(net.ins, net.outs, net.recurrent, net.id, dna, False, False, 999999999)

        print(net.dna)

        inputs = []
        next_input = ""

        while len(inputs) < net.ins:
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.KEYDOWN:
                    if event.unicode.isnumeric() or event.unicode == ".":
                        next_input += event.unicode
                    if event.key == pygame.K_SPACE:
                        inputs.append(float(next_input))
                        next_input = ""
                if event.type == pygame.QUIT:
                    exit()

        net.feed_forward(inputs)

        return net

    def text(self, text, x, y):
        text = self.font.render(text, True, (0, 0, 0))
        self.win.blit(text, (x, y))

    def get_config_options(self):
        pass

    def save(self, nets, name):
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, 'data/' + name + '.txt')
        save_file = open(filename, "w")
        save_file.close()
        save_file = open(filename, "a")
        for i in nets:
            save_file.write(json.dumps(i.dna))
            save_file.write("\n")
        save_file.close()

    def load(self, name):
        to_return = []
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, 'data/' + name + '.txt')
        f = open(filename, "r")
        for l in f:
            to_return.append(json.loads(l))
        f.close()
        return to_return
