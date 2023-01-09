import networkx as nx
import matplotlib.pyplot as plt

# Defining a Class
class GraphVisualization:

    def __init__(self):
        # visual is a list which stores all
        # the set of edges that constitutes a
        # graph
        self.visual = []
        self.nodes = []

    # addEdge function inputs the vertices of an
    # edge and appends it to the visual list
    def addNode(self, id, pos):
        self.nodes.append((id, pos))

    def addEdge(self, a, b):
        temp = [a, b]
        self.visual.append(temp)

    # In visualize function G is an object of
    # class Graph given by networkx G.add_edges_from(visual)
    # creates a graph with a given list
    # nx.draw_networkx(G) - plots the graph
    # plt.show() - displays the graph
    def visualize(self):
        print(self.nodes)
        G = nx.Graph()
        for id, pos in self.nodes:
            G.add_node(id, pos=pos.tolist())
        pos = nx.get_node_attributes(G, 'pos')
        G.add_edges_from(self.visual)
        nx.draw(G, pos)
        plt.show()


