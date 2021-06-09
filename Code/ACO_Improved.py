#!/usr/bin/env python

# from aco import Map
# from aco import Colony
import numpy as np
import sys
import argparse
from grid import grid
import math
import time
#!/usr/bin/env python

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

all_best_fitness = []
all_best_fitness_paths = []

map_dim = (10,10)
start_node = (0,0)
end_node = (9,9)
# obstacles = [(1,2),(1,3)]
obstacles = [(6, 6), (8, 3), (0, 1), (3, 1), (6, 1), (9, 4), (0, 8), (2, 5), (1, 9), (8, 2), (7, 3), (9, 3), (6, 2), (7, 6), (7, 2), (0, 2), (4, 1), (6, 3), (3, 2), (1, 8), (6, 4), (2, 2), (8, 1), (1, 2), (8, 6), (8, 7), (4, 2), (8, 8), (0, 7)]



# map_dim = (31,31)
# start_node = (0,0)
# end_node = (29,30)
# obstacles = [(0, 10), (0, 11), (0, 12), (0, 13), (0, 14), (0, 20), (0, 21), (0, 22), (0, 23), (0, 24), (0, 25), (0, 26), (1, 10), (1, 11), (1, 12), (1, 13), (1, 14), (2, 2), (2, 3), (2, 4), (2, 5), (2, 29), (2, 30), (3, 2), (3, 3), (3, 4), (3, 5), (3, 20), (3, 21), (3, 24), (3, 25), (3, 26), (3, 29), (3, 30), (4, 10), (4, 11), (4, 12), (4, 13), (4, 14), (4, 20), (4, 21), (4, 24), (4, 25), (4, 26), (5, 0), (5, 1), (5, 2), (5, 6), (5, 7), (5, 12), (5, 13), (5, 14), (5, 20), (5, 21), (5, 24), (5, 25), (5, 26), (6, 0), (6, 1), (6, 2), (6, 6), (6, 7), (6, 10), (6, 13), (6, 14), (6, 20), (6, 21), (6, 24), (6, 25), (6, 26), (7, 0), (7, 1), (7, 2), (7, 6), (7, 7), (7, 10), (7, 11), (7, 14), (7, 20), (7, 21), (7, 24), (7, 25), (7, 26), (8, 0), (8, 1), (8, 2), (8, 6), (8, 7), (8, 10), (8, 11), (8, 16), (8, 17), (9, 16), (9, 17), (10, 4), (10, 5), (10, 9), (10, 10), (10, 13), (10, 14), (10, 24), (10, 25), (10, 26), (11, 4), (11, 5), (11, 9), (11, 10), (11, 13), (11, 14), (11, 16), (11, 17), (11, 24), (11, 25), (11, 26), (12, 7), (12, 16), (12, 17), (13, 7), (13, 18), (13, 19), (14, 0), (14, 1), (14, 2), (14, 3), (14, 7), (14, 12), (14, 13), (14, 14), (14, 18), (14, 19), (15, 0), (15, 1), (15, 2), (15, 3), (15, 12), (15, 13), (15, 14), (15, 28), (15, 29), (16, 0), (16, 1), (16, 12), (16, 13), (16, 14), (16, 18), (16, 19), (16, 28), (16, 29), (17, 0), (17, 1), (17, 4), (17, 5), (17, 7), (17, 8), (17, 9), (17, 10), (17, 16), (17, 17), (17, 18), (17, 19), (17, 28), (17, 29), (18, 0), (18, 1), (18, 4), (18, 5), (18, 7), (18, 10), (18, 16), (18, 17), (18, 22), (18, 23), (18, 24), (18, 25), (18, 28), (18, 29), (19, 0), (19, 1), (19, 4), (19, 5), (19, 7), (19, 10), (19, 16), (19, 17), (19, 22), (19, 23), (19, 24), (19, 25), (19, 28), (19, 29), (20, 0), (20, 1), (20, 7), (20, 10), (20, 16), (20, 17), 
# (20, 28), (20, 29), (21, 0), (21, 1), (21, 4), (21, 5), (21, 10), (21, 16), (21, 17), (22, 0), (22, 1), (22, 4), (22, 5), (22, 9), (22, 10), (22, 16), (22, 17), (22, 22), (22, 23), (22, 24), (22, 25), (23, 0), (23, 1), (23, 4), (23, 5), (23, 9), (23, 10), (23, 22), (23, 23), (24, 4), (24, 5), (24, 9), (24, 10), (24, 13), (24, 14), (24, 22), (24, 23), (24, 25), (25, 4), (25, 5), (25, 13), (25, 14), (25, 16), (25, 17), (26, 13), (26, 14), (26, 16), (26, 17), (26, 21), (26, 22), (26, 28), (26, 29), (26, 30), (27, 6), (27, 7), (27, 8), (27, 16), (27, 17), (27, 21), (27, 22), (27, 28), (27, 29), (27, 30), (28, 6), (28, 7), (28, 8), (28, 11), (28, 12), (28, 28), (28, 29), 
# (28, 30), (29, 11), (29, 12), (29, 21), (29, 22), (30, 11), (30, 12), (30, 21), (30, 22)]

# map_dim = (30,30)
# start_node = (0,0)
# end_node = (29,29)
# obstacles = [(4, 17), (15, 4), (4, 13), (5, 13), (28, 8), (29, 25), (21, 21), (18, 17), (27, 4), (7, 9), (21, 26), (28, 13), (13, 22), (9, 18), (17, 22), (7, 14), (3, 27), (26, 5), (20, 18), (16, 14), (23, 1), (16, 10), (3, 15), (21, 23), (22, 23), (1, 24), (16, 15), (25, 19), (7, 24), (27, 2), (21, 15), (17, 11), (3, 20), (13, 24), (26, 15), (15, 7), (27, 28), (25, 11), (27, 11), (2, 4), (7, 29), (10, 29), (0, 8), (28, 20), (14, 12), (16, 29), (6, 4), (14, 8), (9, 17), (26, 4), (25, 4), (7, 23), (26, 11), (27, 3), (3, 21), (2, 24), (15, 8), (17, 12), (26, 2), (21, 20), (26, 3), (25, 3), (26, 10), (13, 12), (16, 22), (3, 24), (4, 20), (16, 7), (2, 15), (25, 2), (29, 20), (10, 18), (5, 12), (6, 3), (27, 10), (26, 6), (10, 17), (21, 22), (0, 7), (20, 21), (3, 4), (24, 11), (6, 5), (5, 17), (28, 3), (4, 4), (3, 16), (19, 21), (28, 21), (4, 21), (25, 5), (11, 18), (4, 16), (24, 2), (4, 18), (2, 3), (5, 5), (10, 19), (1, 7), (6, 24), (29, 8), (15, 5), (28, 14), (8, 24), (8, 9), (8, 18), (24, 12), (7, 18), (22, 24), (4, 5), (14, 9), (26, 16), (20, 20), (23, 2), (14, 24), (6, 14), (16, 12), (6, 9), (9, 29), (9, 9), (4, 24), (3, 17), (3, 26), (8, 25), (15, 14), (14, 22), (20, 26), (2, 14), (19, 26), (4, 6), (22, 1), (8, 23), (22, 20), (25, 10), (26, 1), (5, 24), (5, 16), (16, 4), (23, 0), (25, 15), (17, 17), (29, 13), (2, 7), (1, 8), (13, 11), (17, 18), (22, 2), (18, 11), (23, 3), (27, 12), (9, 28), (4, 12), (18, 12), (5, 23), (26, 17), (22, 0), (3, 13), (24, 3), (25, 17), (8, 29), (8, 19), (9, 23), (19, 18), (26, 18), (13, 10), (26, 9), (3, 12), (28, 12), (26, 28), (21, 1), (15, 9), (21, 24), (10, 23), (23, 23), (5, 4), (25, 18), (0, 6), (15, 15), (7, 25), (13, 13), (15, 6), (26, 19), (3, 19), (8, 26), (6, 2), (7, 28), (16, 13), (12, 24), (29, 14), (24, 10), (20, 19), (17, 10), (3, 18), (4, 27), (25, 12), (28, 28), (28, 2), (14, 10), (25, 9), (16, 23), (9, 19), (5, 14), (4, 22), (15, 13), (6, 12), (23, 20), (3, 3), (14, 11), (6, 16), (14, 5), (19, 27), (28, 1), (6, 11), (4, 7), (22, 25), (7, 17), (4, 19), (9, 16), (25, 16), (13, 8), (29, 9), (16, 11), (7, 10), (22, 21), (9, 22), (16, 8), (4, 23), (20, 22), (24, 19), (17, 7), (18, 18), (17, 23), (25, 14), (16, 9), (6, 15), (19, 19), (26, 12), (22, 22), (27, 5), (6, 18), (7, 26), (23, 25), (16, 16), (17, 13), (3, 2), (28, 4), (28, 22), (2, 23), (6, 23), (1, 6), (20, 23), (8, 10), (20, 27), (7, 5), (1, 23), (23, 19), (6, 28), (2, 26), (15, 24), (28, 10), (2, 17), (23, 4), (27, 9), (15, 23), (11, 17), (5, 15), (19, 11), (23, 21), (28, 11), (5, 3), (29, 12), (7, 2), (9, 15), (4, 25), (12, 18), (23, 22), (2, 5), (15, 25), (19, 12), (24, 15), (20, 12), (10, 15), (10, 14), (17, 16), (7, 16), (25, 8), (3, 7), (20, 1), (3, 14), (12, 22), (2, 22), (7, 19), (23, 18), (6, 10), (15, 12), (20, 0), (5, 6), (6, 27), (19, 20), (24, 5), (22, 26)]

# obstacles = [(6, 6), (8, 3), (0, 1), (3, 1), (6, 1), (9, 4), (0, 8), (2, 5), (1, 9), (8, 2), (7, 3), (9, 3), (6, 2), (7, 6), (7, 2), (0, 2), (4, 1), (6, 3), (3, 2), (1, 8), (6, 4), (2, 2), (8, 1), (1, 2), (8, 6), (8, 7), (4, 2), (8, 8), (0, 7)]
# obstacles = [(9, 11), (14, 6), (16, 14), (7, 15), (4, 1), (19, 10), (5, 14), (2, 3), (11, 9), (3, 1), (17, 14), (3, 2), (18, 14), (18, 15), (6, 14), (4, 0), (19, 15), (2, 4), (5, 1), (13, 6), (14, 5), (6, 15), (6, 13), (4, 2), (5, 15), (16, 13), (17, 15), (8, 15), (4, 15), (2, 2), (6, 1), (13, 5), (12, 6), (5, 0), (15, 13), (3, 3), (11, 10), (19, 14), (16, 15), (7, 16), (18, 13), (7, 14), (7, 17), (8, 14), (13, 7), (1, 1), (1, 2), (14, 7), (5, 13), (15, 5), (12, 5), (18, 10), (6, 0), (4, 14), (17, 10), (7, 18), (8, 18), (14, 13), (2, 5), (7, 13), (15, 4), (12, 10), (16, 5), (3, 0), (19, 13), (2, 1), (18, 16), (17, 13), (15, 6), (1, 3), (14, 14), (11, 8), (11, 5), (15, 15), (9, 10), (15, 14), (17, 9), (10, 9), (18, 11), (11, 11), (10, 11), (14, 12), (13, 13), (12, 7), (3, 5), (12, 11), (1, 4), (18, 17), (9, 12), (7, 0), (13, 14), (16, 16), (10, 10), (13, 8), (6, 18), (18, 12), (9, 15), (12, 14), (8, 12), (8, 11), (15, 12), (7, 1), (13, 15), (7, 19), (6, 17), (6, 12), (3, 4), (15, 7), (15, 11), (10, 5), (7, 12), (18, 18), (11, 7), (9, 9), (5, 16), (6, 16), (12, 8)]
# start_node = (0,0)
# end_node = (19,19)
# map_dim = (20,20)

# map_dim = (6,6)
# start_node = (0,0)
# end_node = (2,3)
# obstacles = [(1,2),(1,3)]

class Map:
    ''' Class used for handling the
        information provided by the
        input map '''
    class Nodes:
        ''' Class for representing the nodes
            used by the ACO algorithm '''
        def __init__(self, row, col, edges):
            self.node_pos= (row, col)
            self.edges = edges

    def __init__(self, alpha, initial_pheromone):
        self.initial_node = start_node
        self.final_node = end_node
        obs = []
        for i in obstacles:
            obs.append((i[1],i[0]))
        self.grid = grid(map_dim[0], obs)
        self.matrix = self.grid.get_matrix()
        # print(len(self.grid[0]))
        # self.grid.visualize([(0,0)])
        self.alpha = alpha
        self.initial_pheromone = initial_pheromone
        self.nodes_array = []
        self.create_nodes(map_dim[0])
    
    def create_nodes(self, dim):
        for row in range(dim):
            nodes_row = []
            for col in range(dim):
                node = self.create_node(row, col)
                nodes_row.append(node)
            self.nodes_array.append(nodes_row)

    def create_node(self, row, col):
        edges = []
        from_edge_on_grid = row*self.grid.dim + col
        for edge_x in range(-1,2,1):
            for edge_y in range(-1,2,1):

                if edge_x != 0 or edge_y!= 0:
                    current_edge_node = (row + edge_x, col + edge_y)
                    to_edge_on_grid = current_edge_node[0]*self.grid.dim + current_edge_node[1]

                    if (current_edge_node[0] >= 0 and current_edge_node[1]>= 0) and (current_edge_node[0] < map_dim[0] and current_edge_node[1] < map_dim[1]):
                        if self.matrix[from_edge_on_grid][to_edge_on_grid] != -1:
                            edges.append({'FinalNode':current_edge_node, 'Pheromone': self.initial_pheromone, 'Probability': 0.0, 'Alpha':self.alpha, 'old_ep': 1})
        
        return self.Nodes(row, col, edges)

    def represent_map(self):
        ''' Represents the map '''
        # Map representation
        plt.plot(self.initial_node[1],self.initial_node[0], 'ro', markersize=10)
        plt.plot(self.final_node[1],self.final_node[0], 'bo', markersize=10)
        nodes = []
        for i in range(map_dim[0]):
            row = []
            for j in range(map_dim[1]):
                row.append(255)
            nodes.append(row)
        for i in obstacles:
            nodes[i[0]][i[1]] = 0
        plt.imshow(nodes, cmap='gray', interpolation = 'nearest', vmin=0, vmax=255)
        plt.show()
        plt.close()

    def represent_path(self, path):
        ''' Represents the path in the map '''
        x = []
        y = []
        for p in path:
            x.append(p[1])
            y.append(p[0])
        plt.plot(x,y)
        self.represent_map()



class Colony:
    class Ant:
        def __init__(self, start, end):
            self.start_node = start
            self.current_node= start
            self.final_node = end
            self.visitedNodes = []
            self.final_node_reached = False
            self.visitedNodes.append(start)

        def move_ant(self, node):
            ''' Moves ant to the selected node '''
            # Compute the total sumatory of the pheromone of each edge
            pheromone_total = 0.0
            for edge in node.edges:
                # if edge['FinalNode'] not in self.visitedNodes:
                pheromone_total += edge['Pheromone']

            # Calculate probability of each edge
            temp_edges = []
            probability = []
            for edge in node.edges:
                # if edge['FinalNode'] not in self.visitedNodes:
                edge['Probability'] = (edge['Pheromone'])/pheromone_total
                temp_edges.append(edge)
                probability.append((edge['Pheromone'])/pheromone_total)

            # Clear probability values
            for edge in node.edges:
                edge['Probability'] = 0.0
            # print(temp_edges)
            # Return the node based on the probability of the solutions
            self.current_node = np.random.choice(temp_edges, 1, probability)[0]['FinalNode']
            self.visitedNodes.append(self.current_node)

        def reset(self):
            ''' Clears the list of visited nodes
                it stores the first one
                and selects the first one as initial'''
            self.visitedNodes[1:] =[]
            self.current_node= self.start_node

    def __init__(self, in_map, no_ants, iterations, evaporation_factor,
                 pheromone_adding_constant):
        self.map = in_map
        self.no_ants = no_ants
        self.iterations = iterations
        self.evaporation_factor = evaporation_factor
        self.pheromone_adding_constant = pheromone_adding_constant
        self.paths = []
        self.ants = self.generate_ants()
        self.best_fitness = []
        self.min_pheromone = 1.0
        self.path_changed = True

    def generate_ants(self):
        ''' Creates a list containin the
            total number of ants specified
            in the initial node '''
        ants = []
        for i in range(self.no_ants):
            ants.append(self.Ant(self.map.initial_node, self.map.final_node))
        return ants

    def update_pheromone_levels(self):
        ''' Updates the pheromone level
            of the each of the trails
            and sorts the paths by lenght '''
        # Sort the list according to the size of the lists
        self.sort_paths()
        
        # Find minimum pheromone
        # min_pheromone = float('inf')

        # for nodes_row in self.map.nodes_array:
        #     for node in nodes_row:
        #         for edge in node.edges:
        #             min_pheromone = min(edge['Pheromone'], min_pheromone)
        
        next_min_pheromone = self.min_pheromone
        
        for path in self.paths:
            for i in range(len(path)):
                for edge in self.map.nodes_array[path[i][0]][path[i][1]].edges:
                    if (i+1) < len(path):

                        # print("edge pher =", edge['Pheromone'], "min_pher = ", min_pheromone)
                        # print(math.exp(min_pheromone / edge['Pheromone']) + math.exp(-1))

                        next_min_pheromone = min(edge['Pheromone'], next_min_pheromone)

                        if self.path_changed:
                            evaporation_factor = edge['old_ep']
                        else:
                            evaporation_factor = math.exp(self.min_pheromone / edge['Pheromone']) + math.exp(-1)
                            edge['old_ep'] = evaporation_factor

                        if edge['FinalNode'] == path[i+1]:
                            edge['Pheromone'] = (evaporation_factor) * \
                            edge['Pheromone'] + self.pheromone_adding_constant/float(len(path))
                        else:
                            edge['Pheromone'] = (evaporation_factor) * edge['Pheromone']

        self.min_pheromone = next_min_pheromone

    def sort_paths(self):
        ''' Sorts the paths '''
        self.paths.sort(key=len)

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

    def delete_loops(self, in_path):
        ''' Checks if there is a loop in the
            resulting path and deletes it '''
        res_path = list(in_path)
        for element in res_path:
            coincidences = self.get_coincidence_indices(res_path, element)
            # reverse de list to delete elements from back to front of the list
            coincidences.reverse()
            for i,coincidence in enumerate(coincidences):
                if not i == len(coincidences)-1:
                    res_path[coincidences[i+1]:coincidence] = []

        return res_path

    def run_colony(self):
        ''' Carries out the process to
            get the best path '''
        # Repeat the cicle for the specified no of times
        for i in range(self.iterations):
            for ant in self.ants:
                ant.reset()
                while not ant.final_node_reached:
                    # Move ant to the next node randomly selected
                    ant.move_ant(self.map.nodes_array[int(ant.current_node[0])][int(ant.current_node[1])])
                    
                    # print(ant.visitedNodes)

                    # Check if solution has been reached
                    if ant.current_node == ant.final_node:
                        ant.final_node_reached = True
                
                # Add the resulting path to the path list
                self.paths.append(self.delete_loops(ant.visitedNodes))

                # Enable the ant for a new search
                ant.final_node_reached = False

            # Update the global pheromone level
            self.update_pheromone_levels()
            self.best_fitness = self.paths[0]

            if len(all_best_fitness) > 0 and all_best_fitness[-1] != len(self.best_fitness):
                self.path_changed = True
            else:
                self.path_changed = False
            
            all_best_fitness.append(len(self.best_fitness))
            

            print ('Iteration: ', i, ' Best path length:', len(self.best_fitness))
        return self.best_fitness


def arguments_parsing():
    ''' Function used for handling the command line argument options '''
    parser = argparse.ArgumentParser()
    parser.add_argument('ants', help = 'the number of ants that made up the \
                        colony', type = int)
    parser.add_argument('iterations', help = 'the number of iterations to be \
                        perfomed by the algorithm', type = int)
    parser.add_argument('map', help = 'the map to calculate the path from', \
                        type = str)
    parser.add_argument('p', help = 'controls the amount of pheromone that is \
                        evaporated, range[0-1], precision 0.05', \
                        type = float, choices = \
                        np.around(np.arange(0.0,1.05,0.05),decimals = 2))
    parser.add_argument('Q', help = 'controls the amount of pheromone that is \
                        added', \
                        type = float)
    parser.add_argument('-d','--display', default = 0, action = 'count', \
                        help = 'display the map and the \
                        resulting path')
    args = parser.parse_args()
    return args.ants, args.iterations, args.map, args.p, args.Q, args.display

def main(print_):
    # ants, iterations, map_path, p, Q, display = arguments_parsing()
    ants = 20
    iterations = 50
    p = 0.05
    Q = 1
    alpha = 1.2
    initial_pheromone = 1.5
    # Get the map
    current_map = Map(alpha=alpha, initial_pheromone=initial_pheromone)
    tic = time.perf_counter()
    ACO = Colony(current_map, ants, iterations, p, Q)
    path = ACO.run_colony()
    toc = time.perf_counter()
    print(f"Run time: {toc - tic:0.4f} seconds")
    
    # plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    # plt.gca().yaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.xlabel('Iterations')
    plt.ylabel('Best Fitness')
    if print_:
        plt.plot(all_best_fitness)
        plt.show()
    print(path)
    if print_:
        current_map.represent_path(path)
    return all_best_fitness

# main(True)