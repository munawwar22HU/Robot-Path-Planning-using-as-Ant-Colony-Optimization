import numpy as np
import matplotlib.pyplot as plt

class grid:
    def __init__(self, dim, obstacles, additional_obstacles = []):
        ''' Initialize the components of grid '''
        self.dim = dim
        self.obstacles = np.array(obstacles)
        self.additional_obstacles = np.array(additional_obstacles)
        if len(self.obstacles.shape) == 1:
            self.obstacles = np.array([self.obstacles%self.dim, self.obstacles//self.dim]).transpose()
        if len(additional_obstacles) > 0 and len(self.additional_obstacles.shape) == 1:
            self.additional_obstacles = np.array([self.additional_obstacles%self.dim, self.additional_obstacles//self.dim]).transpose()

    def get_matrix(self, use_additional=False):
        ''' Generate the adjacency matrix for the grid '''
        matrix = np.ones((self.dim**2, self.dim**2))*(-1)
        for i in range(self.dim**2):
            coordinate = (i%self.dim, i//self.dim)
            if coordinate[0] != 0:
                matrix[i][i-1] = 1                  # left
            if coordinate[0] != self.dim-1:
                matrix[i][i+1] = 1                  # right
            if coordinate[1] != 0:
                matrix[i][i-self.dim] = 1           # top
            if coordinate[1] != self.dim-1:
                matrix[i][i+self.dim] = 1           # bottom
            if coordinate[0] != 0 and coordinate[1] != 0:
                matrix[i][i-self.dim-1] = 1         # top-left
            if coordinate[0] != self.dim-1 and coordinate[1] != 0:
                matrix[i][i-self.dim+1] = 1         # top-right
            if coordinate[0] != 0 and coordinate[1] != self.dim-1:
                matrix[i][i+self.dim-1] = 1         # bottom-left
            if coordinate[0] != self.dim-1 and coordinate[1] != self.dim-1:
                matrix[i][i+self.dim+1] = 1         # bottom-right
        
        if self.obstacles.shape[0] > 0:
            obstacles = self.obstacles[:,0] + self.obstacles[:,1]*self.dim
            for i in obstacles:
                matrix[i,:] = -1
                matrix[:,i] = -1
        if use_additional and self.additional_obstacles.shape[0] > 0:
            additional_obstacles = self.additional_obstacles[:,0] + self.additional_obstacles[:,1]*self.dim
            for i in additional_obstacles:
                matrix[i,:] = -1
                matrix[:,i] = -1

        return matrix
    
    def visualize(self, path, use_additional=False):
        ''' Visualize the grid using the given path '''
        path = np.array(path)
        if len(path.shape) == 1:
            if path.shape[0] > 2:
                plt.plot(path%self.dim, path//self.dim)
            plt.plot(path[0]%self.dim, path[0]//self.dim, 'ro', markersize=10)
            plt.plot(path[-1]%self.dim, path[-1]//self.dim, 'bo', markersize=10)
        else:
            if path.shape[0] > 2:
                plt.plot(path[:,0], path[:,1])
            plt.plot(path[0][0], path[0][1], 'ro', markersize=10)
            plt.plot(path[-1][0], path[-1][1], 'bo', markersize=10)
        nodes = np.ones((self.dim,self.dim, 3))
        for i in self.obstacles:
            nodes[i[1]][i[0]][0] = 0
            nodes[i[1]][i[0]][1] = 0
            nodes[i[1]][i[0]][2] = 0
        if use_additional:
            for i in self.additional_obstacles:
                nodes[i[1]][i[0]][0] = 1
                nodes[i[1]][i[0]][1] = 0
                nodes[i[1]][i[0]][2] = 0
        plt.imshow(nodes, interpolation='nearest', vmin=0, vmax=255)
        plt.show()