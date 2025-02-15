import networkx as nx

class GNMInstance:
    def __init__(self,no_nodes, no_edges, seed=None):
        self.no_nodes = no_nodes
        self.no_edges = no_edges

        self._graph = None
        self._adjacency = None

    def _generate(self):
        self._graph = nx.gnm_random_graph(self.no_nodes, self.no_edges)

    @property
    def n(self):
        return self.no_nodes

    @property
    def m(self):
        return self.no_edges

    @property
    def adjacency(self):
        if self._adjacency is not None:
            return self._adjacency

        self._adjacency = nx.to_numpy_array(self.graph)
        return self._adjacency

    @property
    def graph(self):
        if self._graph is None:
            self._generate()

        return self._graph


class GraphInstance:
    def __init__(self, G: nx.Graph):
        self.no_nodes = G.number_of_nodes()
        self.no_edges = G.number_of_edges()

        self._graph = G
        self._adjacency = None

    @property
    def n(self):
        return self.no_nodes

    @property
    def m(self):
        return self.no_edges

    @property
    def adjacency(self):
        if self._adjacency is not None:
            return self._adjacency

        self._adjacency = nx.to_numpy_array(self.graph)
        return self._adjacency

    @property
    def graph(self):
        return self._graph


# The 4 smallest asymmetric graphs

ASYMMETRIC_GRAPH_1 = nx.Graph()
ASYMMETRIC_GRAPH_1.add_nodes_from(range(6))
ASYMMETRIC_GRAPH_1.add_edges_from(
    [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (1, 5),
        (5, 2),
    ]
)

ASYMMETRIC_GRAPH_2 = nx.Graph()
ASYMMETRIC_GRAPH_2.add_nodes_from(range(6))
ASYMMETRIC_GRAPH_2.add_edges_from(
    [
        (0, 1),
        (1, 3),
        (0, 2),
        (2, 3),
        (2, 4),
        (3, 5),
        (3, 4),
    ]
)

ASYMMETRIC_GRAPH_3 = nx.Graph()
ASYMMETRIC_GRAPH_3.add_nodes_from(range(6))
ASYMMETRIC_GRAPH_3.add_edges_from(
    [
        (0, 1),
        (0, 2),
        (1, 2),
        (1, 3),
        (2, 4),
        (3, 4),
        (3, 5),
    ]
)

ASYMMETRIC_GRAPH_4 = nx.Graph()
ASYMMETRIC_GRAPH_4.add_nodes_from(range(6))
ASYMMETRIC_GRAPH_4.add_edges_from(
    [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 3),
        (2, 3),
        (2, 4),
        (3, 5),
    ]
)