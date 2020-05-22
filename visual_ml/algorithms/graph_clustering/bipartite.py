import numpy as np

from graph_factory import Graph
from numpy import linalg as LA


class Bipartite:
    def __init__(self, graph: Graph):
        if not isinstance(graph, Graph):
            raise TypeError("graph should be type Graph")
        self.graph = graph

    def min_cut(self):
        """
        Find the best bipartite graph. The solution for the optimization function is the vector that
        corresponds to the second lowest eigen vector.
        :return:
        """
        clusters = np.empty(self.graph.size())
        eigen_values, eigen_vectors = LA.eig(self.graph.laplacian_matrix())

        for index, val in enumerate(eigen_vectors[1]):
            if val < 0:
                clusters[index] = 0
            else:
                clusters[index] = 1

        return clusters


if __name__ == '__main__':
    g = Graph(7)

    g.add_edge(0, 1, 0.6)
    g.add_edge(0, 3, 0.8)
    g.add_edge(0, 4, 0.1)
    g.add_edge(1, 2, 0.5)
    g.add_edge(2, 3, 0.7)
    g.add_edge(3, 5, 0.2)
    g.add_edge(4, 5, 0.9)
    g.add_edge(4, 6, 0.7)
    g.add_edge(5, 6, 0.8)

    print(g.adj_matrix())

    min_cut = Bipartite(g)
    print(min_cut.min_cut())
