import pygame
import os
import shutil
import json
from network import Network
import copy
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import csv
import inspect
import math


class DataConf:

    def __init__(self, win):
        self.win = win
        self.font = pygame.font.SysFont("comicsansms", 12)
        self.scores_path = os.path.join(os.path.dirname(inspect.stack()[1].filename), "data")
        self.save_path =  os.path.join(os.path.dirname(__file__), "saves")
        self.data_path =  os.path.join(os.path.dirname(__file__), "data")
        self.tot_saves = len([name for name in os.listdir(self.save_path) if os.path.isfile(os.path.join(self.save_path, name))]) - 1
        self.options = self.set_opts()
        self.gen = 0



    def set_opts(self):
        options = dict()
        filename = os.path.join(self.data_path, 'config.txt')
        config_file = open(filename, "r")
        for line in config_file:
            tmp = line.split()
            options[tmp[0]] = tmp[1]
        return options

    def get_opt(self, option):
        return self.options[option]


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
            self.text(string, 50, 10, (0,0,0))

            pygame.display.update()

    def clear_saves(self):
        folder = dirname = self.save_path
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

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
                self.text(str(neurons[n][m].output_value), x + 22, y + 25, (0,0,0))

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

    def text(self, text, x, y, col):
        text = self.font.render(text, True, col)
        self.win.blit(text, (x, y))

    def get_config_options(self):
        pass

    def save(self, nets):
        name = "generation" + str(self.gen)
        filename = os.path.join(self.save_path, name + '.txt')
        save_file = open(filename, "w")
        save_file.close()
        save_file = open(filename, "a")
        for i in nets:
            try:
                save_file.write(str(i.id) + "\n")
                save_file.write("s" + str(i.species) + "\n")
                save_file.write(json.dumps(i.dna))
                save_file.write("\n")
                save_file.write("-")
                save_file.write(json.dumps(i.start_loc))
                save_file.write("\n")
            except Exception as e:
                print(e)
                print(i.dna)
        save_file.close()

    def set_chosen(self, chosen):
        name = "generation" + str(self.gen)
        filename = os.path.join(self.save_path, name + '.txt')
        save_file = open(filename, "a")
        save_file.write("~")
        save_file.write(json.dumps(chosen))
        save_file.close()

    def load(self):
        name = "generation" + str(self.gen)
        to_return = {"dna": [], "id": [], "chosen" : {}, "loc" : [], "species" : []}
        filename = os.path.join(self.save_path, name + '.txt')
        f = open(filename, "r")
        for l in f:
            try:
                if l[0] == "{":
                    loaded = json.loads(l)
                    to_return["dna"].append(loaded)
                elif l[0] == "~":
                    to_return["chosen"] = json.loads(l[1:])
                elif l.strip().isdigit():
                    to_return["id"].append(int(l))
                elif l[0] == "-":
                    loaded = json.loads(l[1:])
                    to_return["loc"].append(loaded)
                elif l[0] == "s":
                    to_return["species"].append(int(l[1:]))

            except Exception as e:
                print(e)
        f.close()
        return to_return

    def create_data_file(self, labels):
        with open(os.path.join(self.scores_path, 'scores.csv'), mode='w', newline='') as csv_file:
            csv_file = csv.writer(csv_file, delimiter=',')
            csv_file.writerow(labels + ["Generation"])

    def save_data(self, data):
        try:
            with open(os.path.join(self.scores_path, 'scores.csv'), mode='a', newline='') as csv_file:
                csv_file = csv.writer(csv_file, delimiter=',')
                csv_file.writerow(data + [self.gen])
        except Exception as e:
            print(e)

    def change_gen(self, num, add):
        if add:
            self.gen += num
        else:
            self.gen = num

        if self.gen < 0:
            self.gen = 0
            return False
        if add and self.gen > self.tot_saves:
            self.gen = self.tot_saves
            return False
        return True

    def show_graph(self):
        self.axs_lines = {}
        plt.rcParams['figure.figsize'] = [20, 30]
        plt.rcParams["figure.dpi"] = 80
        filename = os.path.join(self.scores_path, 'scores.csv')
        self.dat = {}
        maxes = {}
        mins = {}
        self.i_trans = {}
        colours = ('b','g','r','c','m','y','k','w')

        with open(filename, "r") as csvfile:
            plots = csv.reader(csvfile, delimiter=',')
            for c, row in enumerate(plots):
                for i in range(len(row)):
                    if c == 0:
                        self.i_trans = {h:row[h] for h in range(len(row))}
                        self.dat = {h:[] for h in row}
                        maxes = {h:0 for h in row}
                        mins = {h:float('inf') for h in row}
                    else:
                        trans = self.i_trans[i]
                        if c > self.gen: break
                        self.dat[trans].append(int(float(row[i])))
                        if self.dat[trans][-1] > maxes[trans]:
                            maxes[trans] = self.dat[trans][-1]
                        if self.dat[trans][-1] < mins[trans]:
                            mins[trans] = self.dat[trans][-1]

        if len(self.dat["Generation"]) == 0:
            return

        graphs = (len(self.dat) - 1) ** 0.5
        remainder = graphs - int(graphs)
        rws = int(graphs)
        cols = int(graphs)
        if remainder > 0.5:
            rws += 1
            cols += 1
        elif remainder > 0:
            rws += 1

        fig, self.axs = plt.subplots(rws, cols, constrained_layout=True)


        for j in range(rws):
            for i in range(cols):
                idx = j * cols + i
                if idx >= len(self.dat) or self.i_trans[idx] == "Generation":
                    self.axs[j,i].remove()
                else:
                    trans = self.i_trans[idx]
                    self.axs_lines[trans] = self.axs[j,i].plot([],[], label=trans, lw=3, color=colours[idx], alpha=0.8)[0]
                    self.axs[j,i].set_title(trans, fontsize=18)
                    self.axs[j,i].set_ylim(mins[trans] * 0.8, maxes[trans] * 1.2)
                    self.axs[j,i].set_xlim(0, len(self.dat["Generation"]) - 1)
                    self.axs[j,i].grid()
        step = math.floor(self.gen/100) + 1

        anim = ani.FuncAnimation(fig, self.animate,  frames=range(0,len(self.dat["Generation"]) + 1, step), interval=1, blit=True, repeat=False)
        fig.canvas.draw()
        plt.show()


    def animate(self, iter):
        for i in range(0, len(self.dat)):
            if self.i_trans[i] == "Generation": continue
            self.axs_lines[self.i_trans[i]].set_data(self.dat["Generation"][:iter], self.dat[self.i_trans[i]][:iter])


        return self.axs_lines.values()



