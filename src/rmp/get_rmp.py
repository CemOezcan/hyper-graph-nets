"""
Utility class to select a remote message passing strategy based on a given config file
"""
from src.rmp.gaussian_mixture import GaussianMixtureClustering
from src.rmp.hdbscan import HDBSCAN
from src.rmp.hierarchical_connector import HierarchicalConnector
from src.rmp.multigraph_connector import MultigraphConnector
from src.rmp.random_clustering import RandomClustering
from src.rmp.spectral_clustering import SpectralClustering
from util.Types import *
from src.rmp.remote_message_passing import RemoteMessagePassing
from src.rmp.abstract_clustering_algorithm import AbstractClusteringAlgorithm
from src.rmp.abstract_connector import AbstractConnector

from util.Functions import get_from_nested_dict


def get_rmp(config: ConfigDict) -> RemoteMessagePassing:
    # TODO: Change config template to fit the following
    clustering_name = get_from_nested_dict(config, list_of_keys=["rmp", "clustering"], raise_error=True).lower()
    connector_name = get_from_nested_dict(config, list_of_keys=["rmp", "connector"], raise_error=True).lower()

    clustering = get_clustering_algorithm(clustering_name, config)
    connector = get_connector(connector_name)

    return RemoteMessagePassing(clustering, connector)


def get_clustering_algorithm(name: str, config) -> AbstractClusteringAlgorithm:
    num_clusters = get_from_nested_dict(config, list_of_keys=["rmp", "num_clusters"], raise_error=True)
    sampling = get_from_nested_dict(config, list_of_keys=["rmp", "intra_cluster_sampling", "enabled"], raise_error=True)
    top_k = get_from_nested_dict(config, list_of_keys=["rmp", "intra_cluster_sampling", "top_k"], raise_error=True)
    spotter_threshold = get_from_nested_dict(config, list_of_keys=["rmp", "intra_cluster_sampling", "spotter_threshold"], raise_error=True)
    alpha = get_from_nested_dict(config, list_of_keys=["rmp", "intra_cluster_sampling", "alpha"], raise_error=True)
    hdbscan_spotter_threshold = get_from_nested_dict(config, list_of_keys=["rmp", "hdbscan", "threshold"], raise_error=True)
    hdbscan_max_cluster_size = get_from_nested_dict(config, list_of_keys=["rmp", "hdbscan", "max_cluster_size"], raise_error=True)
    hdbscan_min_cluster_size = get_from_nested_dict(config, list_of_keys=["rmp", "hdbscan", "min_cluster_size"], raise_error=True)

    if name == "hdbscan":
        return HDBSCAN(sampling, hdbscan_max_cluster_size, hdbscan_min_cluster_size, hdbscan_spotter_threshold, top_k)
    elif name == "random":
        return RandomClustering(num_clusters, sampling, spotter_threshold, alpha, top_k)
    elif name == "spectral":
        return SpectralClustering(num_clusters, sampling, spotter_threshold, alpha, top_k)
    elif name == "gmm":
        return GaussianMixtureClustering(num_clusters, sampling, spotter_threshold, alpha, top_k)
    elif name == "none":
        return None
    else:
        raise NotImplementedError("Implement your clustering algorithms here!")


def get_connector(name: str) -> AbstractConnector:
    if name == "hierarchical":
        return HierarchicalConnector()
    elif name == "multigraph":
        return MultigraphConnector()
    elif name == "none":
        return None
    else:
        raise NotImplementedError("Implement your connectors here!")

