import pygame
import os
import json

class DataConf:

	def __init__(self, win):
		self.win = win
		


	def draw_net(self, dna):
		locations = {}
		self.win.fill((255,255,255))
		width, height = pygame.display.get_surface().get_size()

		layers = len(dna["neurons"])
		for i in range(10000):
			for n in range(len(dna["neurons"])):
				num = len(dna["neurons"][n])
				offsetw = width /  layers 
				offseth = height /  num 
				for m in range(len(dna["neurons"][n])):
					x = n * offsetw + offsetw / 2 - 25
					y = (m * offseth + offseth / 2) - 25
					pygame.draw.rect(self.win, (255, 0, 0), (x, y, 50, 50))
					locations[dna["neurons"][n][m]] = (x + 25, y + 25)

		for i in range(len(dna["connections"])):
			connection = dna["connections"][i]
			colour = (255, 0, 0) if connection[2] < 0 else (0, 255, 255)
			if not connection[3]: colour = (200,200,200)
			pygame.draw.line(self.win, colour, (locations[connection[0]][0], locations[connection[0]][1]), (locations[connection[1]][0], locations[connection[1]][1]), 2)

		
		pygame.display.update()



	def get_config_options(self):
		pass



	def save(self,nets, name):
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



