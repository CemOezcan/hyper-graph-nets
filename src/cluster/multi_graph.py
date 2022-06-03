class MultiGraph():

    def __init__(self, graph):
        self._clustering_algorithm = None
        self._node_connector = None
        return

    def run_clustering(self):
        self._clustering_algorithm.run()

    def connect(self):
        self._node_connector.run()
