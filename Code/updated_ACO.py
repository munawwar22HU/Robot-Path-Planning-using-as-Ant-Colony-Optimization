import random
import numpy as np
from grid import grid
import matplotlib.pyplot as plt

class Node:
    def __init__(self, coord, edges, initial_pheramone, alpha):
        ''' Initialize the components of grid '''
        self.coord = coord
        self.edges = edges
        self.alpha = alpha
        self.phermones = np.ones(len(edges), np.float32)*initial_pheramone
        temp = self.phermones**alpha
        self.probability = temp/temp.sum()


class ACO:
    def __init__(self, ants, iterations, evaporation_factor, persistance, initial_pheramone, alpha,  dim, obstacles) -> None:
        ''' Initialize the components of ACO '''
        self.ants = ants
        self.iterations = iterations
        self.evaporation_factor = evaporation_factor
        self.persistance = persistance
        self.alpha = alpha
        self.dim = dim
        self.obstacles = obstacles
        self.grid = []
        self.map = grid(self.dim, self.obstacles)
        for y in range(self.dim):
            row = []
            for x in range(self.dim):
                if (x,y) not in obstacles:
                    edges = []
                    if x != 0 and (x-1, y) not in obstacles:
                        edges.append((x-1, y))                                                  # left
                    if x != self.dim-1 and (x+1, y) not in obstacles:
                        edges.append((x+1, y))                                                  # right
                    if y != 0 and (x, y-1) not in obstacles:
                        edges.append((x, y-1))                                                  # top
                    if y != self.dim-1 and (x, y+1) not in obstacles:
                        edges.append((x, y+1))                                                  # bottom
                    if x != 0 and y != 0 and (x-1, y-1) not in obstacles:
                        edges.append((x-1, y-1))                                                # top-left
                    if x != self.dim-1 and y != 0 and (x+1, y-1) not in obstacles:
                        edges.append((x+1, y-1))                                                # top-right
                    if x != 0 and y != self.dim-1 and (x-1, y+1) not in obstacles:
                        edges.append((x-1, y+1))                                                # bottom-left
                    if x != self.dim-1 and y != self.dim-1 and (x+1, y+1) not in obstacles:
                        edges.append((x+1, y+1))                                                # bottom-right
                    row.append(Node((x,y), edges, initial_pheramone, alpha))
                else:
                    row.append(None)
            self.grid.append(row)

    def generate_path(self, start_node, end_node):
        ''' Generate a path from start node to end node
            based on the pheramone concentration '''
        path = [tuple(start_node)]
        while path[-1] != tuple(end_node):
            prev_node = self.grid[path[-1][1]][path[-1][0]]
            node = tuple(random.choices(prev_node.edges, prev_node.probability)[0])
            path.append(node)
        return self.remove_cycle(path)
    
    def get_coincidence_indices(self,lst, element):
        ''' Gets the indices of the coincidences
            of elements in the path '''
        result = []
        offset = -1
        while True:
            try:
                offset = lst.index(element, offset+1)
            except ValueError:
                return result
            result.append(offset)

    def remove_cycle(self, path):
        ''' Remove cycles (if any) from the given path '''
        res_path = list(path)
        for element in res_path:
            coincidences = self.get_coincidence_indices(res_path, element)
            coincidences.reverse()
            for i, coincidence in enumerate(coincidences):
                if not i == len(coincidences)-1:
                    res_path[coincidences[i+1]:coincidence] = []

        return res_path

    def update_pheramone(self, ant_paths):
        ''' Update the value of pheramone of each edge based on the
            paths constructed by the ants '''
        delta_pheramone = np.zeros((self.dim*self.dim, self.dim*self.dim))
        for path in ant_paths:
            lin_path = np.array(path)
            lin_path = lin_path[:,0] + self.dim*lin_path[:,1]
            for i in range(len(path)-1):
                delta_pheramone[lin_path[i]][lin_path[i+1]] += self.persistance/(len(path)-1)
        for y in range(self.dim):
            for x in range(self.dim):
                if self.grid[y][x] != None:
                    for i in range(len(self.grid[y][x].edges)):
                        (x_, y_) = self.grid[y][x].edges[i]
                        self.grid[y][x].phermones[i] = self.grid[y][x].phermones[i]*(1-self.evaporation_factor) + self.evaporation_factor*delta_pheramone[x+y*self.dim][x_+y_*self.dim]
                    self.grid[y][x].probability = self.grid[y][x].phermones/self.grid[y][x].phermones.sum()

    def get_fitness(self, paths):
        ''' Calculate the fitness of each ant '''
        return [len(path)-1 for path in paths]

    def run(self, start_node, end_node):
        ''' Find the best path from start node to the end node
            while optimizing the length of the path '''
        best_so_far = np.inf
        iter_fitness = []
        
        for iter in range(self.iterations):
            print('Iteration', iter)
            ant_paths = []
            for ant in range(self.ants):
                ant_paths.append(self.generate_path(start_node, end_node))
            fitness = self.get_fitness(ant_paths)
            for i in range(len(fitness)):
                if fitness[i] < best_so_far:
                    best_so_far = fitness[i]
                    path = ant_paths[i]
            best_so_far = min(min(self.get_fitness(ant_paths)), best_so_far)
            iter_fitness.append(best_so_far)
            self.update_pheramone(ant_paths)
        return path

    def visualize(self, path):
        self.map.visualize(path)
            


obstacles10 = [(6, 6), (8, 3), (0, 1), (3, 1), (6, 1), (9, 4), (0, 8), (2, 5), (1, 9), (8, 2), (7, 3), (9, 3), (6, 2), (7, 6), (7, 2), (0, 2), (4, 1), (6, 3), (3, 2), (1, 8), (6, 4), (2, 2), (8, 1), (1, 2), (8, 6), (8, 7), (4, 2), (8, 8), (0, 7)]
obstacles20 = [(9, 11), (14, 6), (16, 14), (7, 15), (4, 1), (19, 10), (5, 14), (2, 3), (11, 9), (3, 1), (17, 14), (3, 2), (18, 14), (18, 15), (6, 14), (4, 0), (19, 15), (2, 4), (5, 1), (13, 6), (14, 5), (6, 15), (6, 13), (4, 2), (5, 15), (16, 13), (17, 15), (8, 15), (4, 15), (2, 2), (6, 1), (13, 5), (12, 6), (5, 0), (15, 13), (3, 3), (11, 10), (19, 14), (16, 15), (7, 16), (18, 13), (7, 14), (7, 17), (8, 14), (13, 7), (1, 1), (1, 2), (14, 7), (5, 13), (15, 5), (12, 5), (18, 10), (6, 0), (4, 14), (17, 10), (7, 18), (8, 18), (14, 13), (2, 5), (7, 13), (15, 4), (12, 10), (16, 5), (3, 0), (19, 13), (2, 1), (18, 16), (17, 13), (15, 6), (1, 3), (14, 14), (11, 8), (11, 5), (15, 15), (9, 10), (15, 14), (17, 9), (10, 9), (18, 11), (11, 11), (10, 11), (14, 12), (13, 13), (12, 7), (3, 5), (12, 11), (1, 4), (18, 17), (9, 12), (7, 0), (13, 14), (16, 16), (10, 10), (13, 8), (6, 18), (18, 12), (9, 15), (12, 14), (8, 12), (8, 11), (15, 12), (7, 1), (13, 15), (7, 19), (6, 17), (6, 12), (3, 4), (15, 7), (15, 11), (10, 5), (7, 12), (18, 18), (11, 7), (9, 9), (5, 16), (6, 16), (12, 8)]
obstacles30 = [(4, 17), (15, 4), (4, 13), (5, 13), (28, 8), (29, 25), (21, 21), (18, 17), (27, 4), (7, 9), (21, 26), (28, 13), (13, 22), (9, 18), (17, 22), (7, 14), (3, 27), (26, 5), (20, 18), (16, 14), (23, 1), (16, 10), (3, 15), (21, 23), (22, 23), (1, 24), (16, 15), (25, 19), (7, 24), (27, 2), (21, 15), (17, 11), (3, 20), (13, 24), (26, 15), (15, 7), (27, 28), (25, 11), (27, 11), (2, 4), (7, 29), (10, 29), (0, 8), (28, 20), (14, 12), (16, 29), (6, 4), (14, 8), (9, 17), (26, 4), (25, 4), (7, 23), (26, 11), (27, 3), (3, 21), (2, 24), (15, 8), (17, 12), (26, 2), (21, 20), (26, 3), (25, 3), (26, 10), (13, 12), (16, 22), (3, 24), (4, 20), (16, 7), (2, 15), (25, 2), (29, 20), (10, 18), (5, 12), (6, 3), (27, 10), (26, 6), (10, 17), (21, 22), (0, 7), (20, 21), (3, 4), (24, 11), (6, 5), (5, 17), (28, 3), (4, 4), (3, 16), (19, 21), (28, 21), (4, 21), (25, 5), (11, 18), (4, 16), (24, 2), (4, 18), (2, 3), (5, 5), (10, 19), (1, 7), (6, 24), (29, 8), (15, 5), (28, 14), (8, 24), (8, 9), (8, 18), (24, 12), (7, 18), (22, 24), (4, 5), (14, 9), (26, 16), (20, 20), (23, 2), (14, 24), (6, 14), (16, 12), (6, 9), (9, 29), (9, 9), (4, 24), (3, 17), (3, 26), (8, 25), (15, 14), (14, 22), (20, 26), (2, 14), (19, 26), (4, 6), (22, 1), (8, 23), (22, 20), (25, 10), (26, 1), (5, 24), (5, 16), (16, 4), (23, 0), (25, 15), (17, 17), (29, 13), (2, 7), (1, 8), (13, 11), (17, 18), (22, 2), (18, 11), (23, 3), (27, 12), (9, 28), (4, 12), (18, 12), (5, 23), (26, 17), (22, 0), (3, 13), (24, 3), (25, 17), (8, 29), (8, 19), (9, 23), (19, 18), (26, 18), (13, 10), (26, 9), (3, 12), (28, 12), (26, 28), (21, 1), (15, 9), (21, 24), (10, 23), (23, 23), (5, 4), (25, 18), (0, 6), (15, 15), (7, 25), (13, 13), (15, 6), (26, 19), (3, 19), (8, 26), (6, 2), (7, 28), (16, 13), (12, 24), (29, 14), (24, 10), (20, 19), (17, 10), (3, 18), (4, 27), (25, 12), (28, 28), (28, 2), (14, 10), (25, 9), (16, 23), (9, 19), (5, 14), (4, 22), (15, 13), (6, 12), (23, 20), (3, 3), (14, 11), (6, 16), (14, 5), (19, 27), (28, 1), (6, 11), (4, 7), (22, 25), (7, 17), (4, 19), (9, 16), (25, 16), (13, 8), (29, 9), (16, 11), (7, 10), (22, 21), (9, 22), (16, 8), (4, 23), (20, 22), (24, 19), (17, 7), (18, 18), (17, 23), (25, 14), (16, 9), (6, 15), (19, 19), (26, 12), (22, 22), (27, 5), (6, 18), (7, 26), (23, 25), (16, 16), (17, 13), (3, 2), (28, 4), (28, 22), (2, 23), (6, 23), (1, 6), (20, 23), (8, 10), (20, 27), (7, 5), (1, 23), (23, 19), (6, 28), (2, 26), (15, 24), (28, 10), (2, 17), (23, 4), (27, 9), (15, 23), (11, 17), (5, 15), (19, 11), (23, 21), (28, 11), (5, 3), (29, 12), (7, 2), (9, 15), (4, 25), (12, 18), (23, 22), (2, 5), (15, 25), (19, 12), (24, 15), (20, 12), (10, 15), (10, 14), (17, 16), (7, 16), (25, 8), (3, 7), (20, 1), (3, 14), (12, 22), (2, 22), (7, 19), (23, 18), (6, 10), (15, 12), (20, 0), (5, 6), (6, 27), (19, 20), (24, 5), (22, 26)]

ants = 40
iterations = 50
evaporation_factor = 0.6
persistance = 1
initial_pheramone = 0.1
alpha = 1.2
dim = 30

aco = ACO(ants, iterations, evaporation_factor, persistance, initial_pheramone, alpha, dim, obstacles30)
path = aco.run((0,0), (dim-1,dim-1))
print('Path:', path)
aco.visualize(path)